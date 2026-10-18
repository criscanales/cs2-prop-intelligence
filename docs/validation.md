# Model validation (rolling windows)

Generated 2026-10-18 from `data/player_maps.csv` (last ~3 months, tiers S/A/B). Each 10-day test window is scored by a
model fit only on data before it (validation = the 10 days before the window for the shrink factor).

## Map pool model

Lower is better. `base` = previous play-frequency blend; `pool` = map pool model; win log loss: `same` = one Elo win
chance on every map, `permap` = per-map strength shift.

```
                window subset  train_series  series  pick_nll_base  pick_nll_pool  brier12_base  brier12_pool  maps  win_ll_same  win_ll_permap
2026-10-08..2026-10-18    all          1146     189         3.9219         3.8372        0.1845        0.1822   441       0.6867         0.6917
2026-10-08..2026-10-18 tier s          1146      21         4.1078         3.9726        0.1953        0.1946    51       0.7121         0.7207
2026-10-08..2026-10-18 tier b          1146     168         3.8987         3.8203        0.1832        0.1806   390       0.6833         0.6879
2026-10-08..2026-10-18    BO1          1146      10         1.8801         1.6876        0.1168        0.1112    10       0.7007         0.7014
2026-10-08..2026-10-18    BO3          1146     177         4.0153         3.9385        0.1880        0.1858   424       0.6862         0.6916
2026-09-28..2026-10-08    all           956     190         3.3296         3.1860        0.1633        0.1559   378       0.6681         0.6638
2026-09-28..2026-10-08 tier s           956      33         3.9715         3.7275        0.1824        0.1696    80       0.6934         0.6999
2026-09-28..2026-10-08 tier a           956      14         4.1693         4.1448        0.1816        0.1810    36       0.6866         0.6691
2026-09-28..2026-10-08 tier b           956     143         3.0992         2.9671        0.1571        0.1502   262       0.6579         0.6521
2026-09-28..2026-10-08    BO1           956      60         1.8563         1.8032        0.1214        0.1179    60       0.6088         0.5988
2026-09-28..2026-10-08    BO3           956     129         3.9966         3.8116        0.1827        0.1735   314       0.6749         0.6709
2026-09-18..2026-09-28    all           702     254         3.9002         3.8347        0.1795        0.1787   606       0.6953         0.6938
2026-09-18..2026-09-28 tier a           702      21         3.9585         3.7651        0.1783        0.1709    53       0.7434         0.7438
2026-09-18..2026-09-28 tier b           702     226         3.8811         3.8244        0.1791        0.1788   535       0.6906         0.6890
2026-09-18..2026-09-28    BO3           702     249         3.8908         3.8255        0.1797        0.1790   591       0.6950         0.6933
2026-09-08..2026-09-18    all           503     199         3.7862         3.6880        0.1799        0.1754   442       0.6865         0.6865
2026-09-08..2026-09-18 tier s           503      28         3.8551         3.8071        0.1779        0.1764    67       0.6774         0.6774
2026-09-08..2026-09-18 tier a           503      54         2.9906         2.8798        0.1544        0.1497    94       0.6784         0.6784
2026-09-08..2026-09-18 tier b           503     117         4.1368         4.0324        0.1921        0.1870   281       0.6913         0.6913
2026-09-08..2026-09-18    BO1           503      27         1.8070         1.7082        0.1181        0.1139    27       0.6852         0.6852
2026-09-08..2026-09-18    BO3           503     170         4.0922         3.9942        0.1898        0.1853   409       0.6859         0.6859
```

Pick order and P(played in maps 1-2) improve in every window and every tier/format subset, so the pick model is on.
The per-map win shift is better in some windows and worse in others, so it stays off (the app only enables it when it wins).

## Shrink toward recent form

MAE / bias in kills or headshots per map; `dev_gain%` = mean Poisson deviance improvement per row x100 (positive = better).

```
== by all ==
        rows  kills_mae_raw  kills_mae_adj  kills_dev_gain%  kills_bias_raw  kills_bias_adj  hs_mae_raw  hs_mae_adj  hs_dev_gain%  hs_bias_raw  hs_bias_adj
all                                                                                                                                                        
all  17656.0          3.512          3.509            0.067           0.076           0.079       1.626       1.624           0.1         0.12        0.081

== by window ==
                                                                rows  kills_mae_raw  kills_mae_adj  kills_dev_gain%  kills_bias_raw  kills_bias_adj  hs_mae_raw  hs_mae_adj  hs_dev_gain%  hs_bias_raw  hs_bias_adj
window                                                                                                                                                                                                             
2026-09-08..2026-09-18 (train 12226, lam_k 0.81, lam_h 0.66)  4181.0          3.510          3.504            0.118           0.153           0.146       1.652       1.651         0.077        0.074        0.018
2026-09-18..2026-09-28 (train 16416, lam_k 0.84, lam_h 0.87)  5770.0          3.490          3.490            0.025           0.076           0.080       1.624       1.622         0.087        0.168        0.147
2026-09-28..2026-10-08 (train 20597, lam_k 0.90, lam_h 0.56)  3590.0          3.510          3.509            0.041           0.053           0.060       1.588       1.583         0.195        0.136        0.088
2026-10-08..2026-10-18 (train 26367, lam_k 0.88, lam_h 0.63)  4115.0          3.546          3.542            0.095           0.018           0.028       1.636       1.636         0.059        0.085        0.045

== by tier ==
         rows  kills_mae_raw  kills_mae_adj  kills_dev_gain%  kills_bias_raw  kills_bias_adj  hs_mae_raw  hs_mae_adj  hs_dev_gain%  hs_bias_raw  hs_bias_adj
tier                                                                                                                                                        
a      1808.0          3.454          3.450            0.069           0.255           0.266       1.586       1.579         0.279        0.212        0.162
b     13702.0          3.525          3.523            0.065           0.026           0.025       1.646       1.644         0.081        0.118        0.078
s      2146.0          3.476          3.475            0.077           0.245           0.269       1.536       1.535         0.068        0.053        0.032

== by sample ==
            rows  kills_mae_raw  kills_mae_adj  kills_dev_gain%  kills_bias_raw  kills_bias_adj  hs_mae_raw  hs_mae_adj  hs_dev_gain%  hs_bias_raw  hs_bias_adj
sample                                                                                                                                                         
<20 maps  5588.0          3.526          3.524            0.006           0.223           0.219       1.639       1.637         0.099        0.094        0.051
20-40     5851.0          3.530          3.528            0.043           0.009           0.008       1.613       1.610         0.122        0.118        0.078
40+       6217.0          3.482          3.479            0.143           0.006           0.022       1.627       1.625         0.081        0.145        0.109

== by gap ==
            rows  kills_mae_raw  kills_mae_adj  kills_dev_gain%  kills_bias_raw  kills_bias_adj  hs_mae_raw  hs_mae_adj  hs_dev_gain%  hs_bias_raw  hs_bias_adj
gap                                                                                                                                                            
<1 kill  14576.0          3.490          3.489            0.008           0.072           0.073       1.605       1.604         0.088        0.104        0.064
1-2       2661.0          3.610          3.598            0.364           0.047           0.057       1.699       1.693         0.159        0.178        0.141
2-4        405.0          3.661          3.658            0.177           0.321           0.355       1.862       1.857         0.135        0.300        0.253
4+          14.0          3.683          3.548            1.190           2.721           3.008       2.724       2.706         0.629        0.664        0.605
```

Deviance improves in every window, tier and sample-size bucket, and MAE never gets worse. The gains are small
(about 0.1%), largest where the raw model moves 1+ kills from the player's recent rate. Headshot bias drops in every
slice; kills bias is unchanged (about +0.25 kills/map on tier S and A maps, ~0 on tier B).
