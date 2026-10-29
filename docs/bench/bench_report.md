| Tier | total_seconds | vs. serial | failed | episodes | Scheduler snapshots | Load events |
| --- | --- | --- | --- | --- | --- | --- |
| serial | 396.726856 | — | 0 | 6 | 0 | 12 |
| pipeline | 231.817723 | 1.7114 | 0 | 6 | 0 | 10 |
| adaptive | 145.506701 | 2.7265 | 0 | 6 | 12 | 2 |

action model: openai_vlm

Methodology alignment: serial↔902.499 / pipeline↔644.439 / adaptive↔536.550 (checklist doc); absolute values are not comparable — hardware and data differ (spec decision D-B3), only the three-tier ratios are comparable within this batch.
