# S04 — Local models and Gself

Values below are copied from S04 artifacts. Gself is an internal portability reference, not physical noise. Families are not ranked.

## Local TEST metrics

| plant | family | n_test | MAE | RMSE | R2 | bias | selected |
|---|---|---:|---:|---:|---:|---:|---|
| PS_002 | HistGradientBoosting | 3527 | 0.069108778 | 0.15273328 | 0.74593984 | 0.027062139 | HistGradientBoosting_06 |
| PS_002 | SplineRidge | 3527 | 0.069910638 | 0.14572901 | 0.76870766 | 0.011342911 | SplineRidge_00 |
| PS_003 | HistGradientBoosting | 3374 | 0.057856889 | 0.1249187 | 0.8213786 | 0.013639706 | HistGradientBoosting_00 |
| PS_003 | SplineRidge | 3374 | 0.059086382 | 0.1204186 | 0.83401618 | 0.0024891569 | SplineRidge_00 |
| PS_035 | HistGradientBoosting | 1628 | 0.060892583 | 0.084792543 | 0.91196756 | 0.04163788 | HistGradientBoosting_01 |
| PS_035 | SplineRidge | 1628 | 0.062763353 | 0.083501959 | 0.91462696 | 0.0360821 | SplineRidge_00 |
| PS_039 | HistGradientBoosting | 1318 | 0.080606557 | 0.1128895 | 0.8588247 | -0.019019764 | HistGradientBoosting_00 |
| PS_039 | SplineRidge | 1318 | 0.096203356 | 0.1264498 | 0.82287169 | -0.036681183 | SplineRidge_02 |
| PS_042 | HistGradientBoosting | 819 | 0.061414004 | 0.08311471 | 0.92092114 | -0.044008059 | HistGradientBoosting_01 |
| PS_042 | SplineRidge | 819 | 0.07604997 | 0.099609233 | 0.88641948 | -0.034260801 | SplineRidge_03 |
| PS_043 | HistGradientBoosting | 791 | 0.08611357 | 0.12007473 | 0.84492353 | -0.034234236 | HistGradientBoosting_07 |
| PS_043 | SplineRidge | 791 | 0.082124782 | 0.11265951 | 0.86348563 | -0.028126409 | SplineRidge_08 |
| PS_044 | HistGradientBoosting | 835 | 0.073586106 | 0.11355818 | 0.85590752 | -0.02506933 | HistGradientBoosting_05 |
| PS_044 | SplineRidge | 835 | 0.074625605 | 0.10827152 | 0.86901157 | -0.032100048 | SplineRidge_07 |
| PS_045 | HistGradientBoosting | 732 | 0.04096804 | 0.075300688 | 0.93619091 | 0.013918607 | HistGradientBoosting_00 |
| PS_045 | SplineRidge | 732 | 0.052267838 | 0.084524223 | 0.91960167 | 0.0093113686 | SplineRidge_08 |
| PS_046 | HistGradientBoosting | 570 | 0.066840687 | 0.10640398 | 0.86506705 | 0.028550272 | HistGradientBoosting_05 |
| PS_046 | SplineRidge | 570 | 0.066591068 | 0.099128047 | 0.88288962 | 0.01592571 | SplineRidge_06 |
| PS_047 | HistGradientBoosting | 576 | 0.12144112 | 0.16530492 | 0.71465902 | -0.095760838 | HistGradientBoosting_05 |
| PS_047 | SplineRidge | 576 | 0.12650181 | 0.16698664 | 0.70882371 | -0.10244918 | SplineRidge_02 |
| PS_048 | HistGradientBoosting | 739 | 0.056599604 | 0.096342966 | 0.87835571 | -0.0019774521 | HistGradientBoosting_00 |
| PS_048 | SplineRidge | 739 | 0.048017728 | 0.090779094 | 0.8920001 | 0.0010466272 | SplineRidge_03 |
| PS_049 | HistGradientBoosting | 710 | 0.05487465 | 0.10647714 | 0.86654723 | 0.0057456492 | HistGradientBoosting_01 |
| PS_049 | SplineRidge | 710 | 0.056863678 | 0.10385558 | 0.87303776 | 0.0056679065 | SplineRidge_03 |
| PS_050 | HistGradientBoosting | 841 | 0.059488904 | 0.10638549 | 0.86779048 | 0.017415371 | HistGradientBoosting_00 |
| PS_050 | SplineRidge | 841 | 0.058046566 | 0.10198542 | 0.87850061 | 0.0030755545 | SplineRidge_02 |
| PS_051 | HistGradientBoosting | 747 | 0.065202281 | 0.14930875 | 0.77686842 | 0.027792505 | HistGradientBoosting_05 |
| PS_051 | SplineRidge | 747 | 0.066535928 | 0.14610659 | 0.7863366 | 0.023017552 | SplineRidge_03 |

## Gself signed mean

| plant | family | B1 | B2 | B3 | mean |
|---|---|---:|---:|---:|---:|
| PS_002 | HistGradientBoosting | 0.0010000734 | 1.5444249e-05 | 0.0004891704 | 0.00050156269 |
| PS_002 | SplineRidge | 0.0011914953 | -0.00064867746 | 0.00067128417 | 0.00040470065 |
| PS_003 | HistGradientBoosting | 0.0010010779 | -0.0001249128 | 0.0008167817 | 0.0005643156 |
| PS_003 | SplineRidge | -0.00025009942 | 0.00055754226 | 0.00070868625 | 0.0003387097 |
| PS_035 | HistGradientBoosting | 0.0026109815 | -0.0013428315 | 0.0044003269 | 0.0018894923 |
| PS_035 | SplineRidge | -0.00054322132 | 0.0012713932 | -0.0018110735 | -0.0003609672 |
| PS_039 | HistGradientBoosting | -0.013111727 | 0.010995972 | 0.017951763 | 0.0052786694 |
| PS_039 | SplineRidge | -0.021462282 | 0.010429841 | 0.014554685 | 0.0011740812 |
| PS_042 | HistGradientBoosting | 0.013186612 | -0.00088631677 | -0.0092809858 | 0.0010064364 |
| PS_042 | SplineRidge | 0.036119847 | -0.0073300066 | -0.018763889 | 0.0033419837 |
| PS_043 | HistGradientBoosting | 0.00087682536 | -0.0019156286 | -0.017288814 | -0.0061092059 |
| PS_043 | SplineRidge | -0.00058943515 | 0.008083354 | -0.0085727183 | -0.00035959981 |
| PS_044 | HistGradientBoosting | 0.0053101626 | -0.0015988976 | 0.00019167313 | 0.0013009794 |
| PS_044 | SplineRidge | -0.0033577052 | 0.0021493166 | 0.00065308097 | -0.00018510256 |
| PS_045 | HistGradientBoosting | 0.0028921566 | 0.0014405267 | 0.0030226238 | 0.002451769 |
| PS_045 | SplineRidge | 0.0037850032 | 0.0078280523 | 0.01756302 | 0.0097253584 |
| PS_046 | HistGradientBoosting | 4.2931413e-05 | 0.00063230054 | 0.00088617663 | 0.00052046953 |
| PS_046 | SplineRidge | -0.0015453779 | -0.0011187347 | 0.0062140619 | 0.0011833164 |
| PS_047 | HistGradientBoosting | 0.0040982609 | -0.0064104269 | 0.027270522 | 0.0083194519 |
| PS_047 | SplineRidge | 0.0032494621 | -0.0059493193 | 0.0079887428 | 0.0017629619 |
| PS_048 | HistGradientBoosting | 0.004803106 | -0.0024899423 | -0.0019416727 | 0.00012383034 |
| PS_048 | SplineRidge | 0.0022897506 | 0.00061274827 | -0.00098862189 | 0.00063795899 |
| PS_049 | HistGradientBoosting | 0.011434107 | 0.0010037879 | 0.011115909 | 0.0078512678 |
| PS_049 | SplineRidge | 0.0026226312 | 0.0005460538 | 0.0096709922 | 0.0042798924 |
| PS_050 | HistGradientBoosting | 0.0011947316 | 0.0011493086 | 0.0015382332 | 0.0012940912 |
| PS_050 | SplineRidge | 0.0070789499 | 0.0012998723 | 0.0041951758 | 0.0041913327 |
| PS_051 | HistGradientBoosting | -0.00032023914 | 0.0018028051 | 0.001004944 | 0.00082916998 |
| PS_051 | SplineRidge | 0.0045604647 | -0.0012599365 | 0.0025278439 | 0.0019427907 |

No common-support, OTG, cross-plant, bootstrap, or permutation results were computed.

**GO — S04 LOCAL MODELS AND GSELF COMPLETE**
