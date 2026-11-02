# SMD profile

Source: original OmniAnomaly ServerMachineDataset (train, test, test_label, interpretation_label). A segment is a run of consecutive 1s in `test_label`.

| Machine | Train rows | Test rows | Anomalous test rows | Anomaly ratio | Anomaly segments | Longest segment | Interpretation lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| machine-1-1 | 28,479 | 28,479 | 2,694 | 9.46% | 8 | 721 | 8 |
| machine-1-2 | 23,694 | 23,694 | 542 | 2.29% | 10 | 156 | 10 |
| machine-1-3 | 23,702 | 23,703 | 817 | 3.45% | 12 | 225 | 11 |
| machine-1-4 | 23,706 | 23,707 | 720 | 3.04% | 12 | 205 | 12 |
| machine-1-5 | 23,705 | 23,706 | 100 | 0.42% | 7 | 31 | 7 |
| machine-1-6 | 23,688 | 23,689 | 3,708 | 15.65% | 30 | 3,161 | 30 |
| machine-1-7 | 23,697 | 23,697 | 2,398 | 10.12% | 13 | 1,215 | 13 |
| machine-1-8 | 23,698 | 23,699 | 763 | 3.22% | 20 | 371 | 20 |
| machine-2-1 | 23,693 | 23,694 | 1,170 | 4.94% | 13 | 452 | 13 |
| machine-2-2 | 23,699 | 23,700 | 2,833 | 11.95% | 11 | 872 | 11 |
| machine-2-3 | 23,688 | 23,689 | 269 | 1.14% | 10 | 91 | 10 |
| machine-2-4 | 23,689 | 23,689 | 1,694 | 7.15% | 20 | 401 | 20 |
| machine-2-5 | 23,688 | 23,689 | 980 | 4.14% | 21 | 371 | 21 |
| machine-2-6 | 28,743 | 28,743 | 424 | 1.48% | 8 | 118 | 8 |
| machine-2-7 | 23,696 | 23,696 | 417 | 1.76% | 20 | 305 | 20 |
| machine-2-8 | 23,702 | 23,703 | 161 | 0.68% | 1 | 161 | 1 |
| machine-2-9 | 28,722 | 28,722 | 1,755 | 6.11% | 10 | 414 | 10 |
| machine-3-1 | 28,700 | 28,700 | 308 | 1.07% | 4 | 131 | 4 |
| machine-3-10 | 23,692 | 23,693 | 1,047 | 4.42% | 13 | 428 | 13 |
| machine-3-11 | 28,695 | 28,696 | 198 | 0.69% | 3 | 126 | 3 |
| machine-3-2 | 23,702 | 23,703 | 1,109 | 4.68% | 10 | 837 | 9 |
| machine-3-3 | 23,703 | 23,703 | 632 | 2.67% | 26 | 481 | 28 |
| machine-3-4 | 23,687 | 23,687 | 977 | 4.12% | 8 | 786 | 8 |
| machine-3-5 | 23,690 | 23,691 | 426 | 1.80% | 11 | 151 | 11 |
| machine-3-6 | 28,726 | 28,726 | 1,194 | 4.16% | 11 | 230 | 11 |
| machine-3-7 | 28,705 | 28,705 | 434 | 1.51% | 5 | 311 | 5 |
| machine-3-8 | 28,703 | 28,704 | 1,371 | 4.78% | 6 | 573 | 6 |
| machine-3-9 | 28,713 | 28,713 | 303 | 1.06% | 4 | 126 | 4 |
| **Total** | **708,405** | **708,420** | **29,444** | **4.16%** | **327** | **3,161** | **327** |

## Summary

SMD has 28 machines with 708,405 train rows and 708,420 test rows, of which 29,444 (4.16%) are labelled anomalous. The anomaly ratio per machine ranges from 0.42% (machine-1-5) to 15.65% (machine-1-6), so a single global threshold will not suit every machine. The anomalies form 327 segments, the longest being 3,161 rows (machine-1-6), and 327 interpretation-label lines are available.
