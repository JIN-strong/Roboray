#!/usr/bin/env bash
# Bring up a KubeRay operator that **watches only this namespace** in the roboray namespace.
#
# Background: the two pre-existing operators on the platform are namespace-scoped
# (one watches --watch-namespace <ns-a>, the other --watch-namespace <ns-b>), and neither
# manages roboray, so CRs created here are never reconciled (no Pods, no Events). This
# script reuses the existing operator's RBAC/image and points watch-namespace at roboray.
#
# Safety: all that is added is a SA/Role/RoleBinding inside roboray, a Deployment,
# and one **new** ClusterRoleBinding (existing bindings are untouched); the operator
# only watches roboray and cannot touch resources in other namespaces.
set -euo pipefail
NS=roboray
# Namespace of the pre-existing operator whose Roles are copied (env-overridable).
ROLE_SRC_NS="${ROLE_SRC_NS:-<operator-source-ns>}"
OP_IMAGE=registry.example.com/kuberay/operator:v1.6.2

K() { sudo kubectl "$@"; }

echo "== 1. ServiceAccount =="
cat <<EOF | K apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kuberay-operator
  namespace: ${NS}
EOF

echo "== 2. Copy the existing operator's Role (incl. leader-election) from ${ROLE_SRC_NS} and re-point it to ${NS} =="
K -n "$ROLE_SRC_NS" get role kuberay-operator kuberay-operator-leader-election -o json \
 | python3 -c "
import json,sys
d=json.load(sys.stdin)
items=d.get('items',[d])
out={'apiVersion':'v1','kind':'List','items':[]}
for it in items:
    it['metadata']={'name':it['metadata']['name'],'namespace':'${NS}'}
    it.pop('status',None)
    out['items'].append(it)
json.dump(out,sys.stdout)
" | K apply -f -

echo "== 3. RoleBinding (one per Role) =="
for r in kuberay-operator kuberay-operator-leader-election; do
cat <<EOF | K apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ${r}
  namespace: ${NS}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: ${r}
subjects:
  - kind: ServiceAccount
    name: kuberay-operator
    namespace: ${NS}
EOF
done

echo "== 4. ClusterRoleBinding (reuse the existing ClusterRole; new binding, old one untouched) =="
cat <<EOF | K apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kuberay-operator-${NS}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kuberay-operator
subjects:
  - kind: ServiceAccount
    name: kuberay-operator
    namespace: ${NS}
EOF

echo "== 5. Deployment (watches only ${NS}) =="
cat <<EOF | K apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kuberay-operator
  namespace: ${NS}
  labels: {app.kubernetes.io/name: kuberay-operator, app.kubernetes.io/instance: roboray}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector:
    matchLabels: {app.kubernetes.io/name: kuberay-operator, app.kubernetes.io/instance: roboray}
  template:
    metadata:
      labels: {app.kubernetes.io/name: kuberay-operator, app.kubernetes.io/instance: roboray}
    spec:
      serviceAccountName: kuberay-operator
      containers:
      - name: kuberay-operator
        image: ${OP_IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["/manager"]
        args:
          - --watch-namespace
          - ${NS}
          - --enable-leader-election=true
          - --reconcile-concurrency=1
          - --qps=100
          - --burst=200
          - --feature-gates=RayClusterStatusConditions=true,RayJobDeletionPolicy=true
        ports: [{containerPort: 8080, name: http}]
        resources: {limits: {cpu: 500m, memory: 512Mi}, requests: {cpu: 100m, memory: 128Mi}}
EOF

echo "== 6. Wait for the operator to be ready =="
K -n ${NS} rollout status deploy/kuberay-operator --timeout=180s
K -n ${NS} get pods
