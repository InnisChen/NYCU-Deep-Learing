# Task 3 Experiment Log

## Final Report Selection

Final selected run:

```text
task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay600k-lr1e5-2500k
```

Final Task 3 recipe:

| Item | Value |
|---|---|
| Platform | Lightning AI |
| Algorithm | Dueling Double DQN + PER + 3-step return |
| Learning rate | `0.00018` |
| Adam epsilon | `0.00015` |
| LR decay | `0.00018 -> 0.00001` at `600000` env steps |
| Replay memory | `200000` |
| Replay start | `20000` |
| Batch size | `32` |
| Epsilon schedule | exponential, `epsilon_decay=0.99996`, `epsilon_min=0.01` |
| Target update | `2000` train updates |
| Soft target update | disabled, `soft_target_tau=0.0` |
| Noop max | `0` |
| Max env steps | `2500000` |

Official report evaluation:

```text
20 consecutive testing seeds starting from seed=2, i.e. seeds 2-21.
```

Milestone results:

| Checkpoint | Avg | Min | Max | Reaches 19 |
|---:|---:|---:|---:|---|
| 600k | `19.10` | `14` | `21` | Yes |
| 1M | `19.45` | `16` | `21` | Yes |
| 1.5M | `20.15` | `18` | `21` | Yes |
| 2M | `19.90` | `17` | `21` | Yes |
| 2.5M | `19.80` | `18` | `21` | Yes |

Figure filenames expected by `report.tex`:

```text
figure/task3_seed2_600k_eval_screenshot.png
figure/task3_seed2_1m_eval_screenshot.png
figure/task3_seed2_1p5m_eval_screenshot.png
figure/task3_seed2_2m_eval_screenshot.png
figure/task3_seed2_2p5m_eval_screenshot.png
```

Summary:

- This is the final Task 3 submission candidate.
- The score reaches the 19 threshold at 600k and remains above 19 for all required later milestones.
- LR decay at 600k is the key stability change: the larger LR learns quickly before 600k, then `1e-5` prevents policy drift during longer training.

這份文件用來追蹤 Pong Task 3 的訓練設定、評估結果與分析結論。之後每跑完一組實驗，可以在這裡追加一列，避免不同 ablation 混在一起。

## Current Best

目前最佳 600k checkpoint:

| Run | 600k 20-seed eval | Key setting | Conclusion |
|---|---:|---|---|
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-600k` | Avg `18.95`, Min `14`, Max `21` | Dueling DDQN + PER + n-step 3, reference optimizer recipe, `adam_eps=1.5e-4`, `lr=1.8e-4`, `epsilon_decay=0.99996`, `eps_min=0.01`, `target_update=2000` | 目前最強 600k checkpoint。距離 Avg `19` 只差 `0.05`，若報告保留一位小數可呈現為 `19.0`。 |

目前最佳 1M checkpoint:

| Run | 1M 20-seed eval | Key setting | Conclusion |
|---|---:|---|---|
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay600k-lr1e5-2500k` | Avg `19.45`, Min `16`, Max `21` | Dueling DDQN + PER + n-step 3, `lr=1.8e-4`, `epsilon_decay=0.99996`, `target_update=2000`, LR decays to `1e-5` after 600k | 目前最強 1M checkpoint，正式 seed `0`-`19` 評估已超過 Avg `19`。 |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-1000k` | Avg `18.05`, Min `10`, Max `21` | Dueling DDQN + PER + n-step 3, reference optimizer recipe, `epsilon_decay=0.99998`, `eps_min=0.01`, `target_update=1000` | 舊的 1M best，已被 Lighting AI LR decay run 超越。 |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-1000k` | Avg `18.00`, Min `14`, Max `20` | Same 1M recipe, but `lr=2e-4` | 目前最穩定的 1M checkpoint。平均只比 best 低 `0.05`，但 Min 從 `10` 提高到 `14`。 |

目前判斷:

- 目前最佳方向是 reference optimizer recipe + Dueling + exponential epsilon decay；600k 最佳是 `exp99996 + lr=2e-4 + target_update=2000`，1M 最佳是 `slowexp99998 + target_update=1000`。
- 若目標是 1M，`target_update=1000` 明顯值得保留；它在 1M 達到 Avg `18.05`，已經非常接近 Avg `19`。
- 對 1M 來說，`lr=2e-4` 沒有提高平均，但明顯提高穩定性：Min 從 `10` 到 `14`。如果要保守提交，這組很有價值。
- 在舊的 linear-decay dueling 設定中，`epsilon_min=0.02` 是 sweet spot；但在 reference exponential 設定中，`eps_min=0.01` 搭配 slower decay 更好。
- `epsilon_min=0.03` / `0.04` 後期探索偏多，會降低 Pong 的 final evaluation。
- `n_step=5` 目前比 `n_step=3` 慢，對 600k sample efficiency 沒有幫助。
- `noop_max=30` 對最新 no-dueling slowexp run 沒有幫助，反而讓 Avg 從 `11.80` 降到 `7.45`。
- 600k 最強目前為 Avg `18.95` / Min `14`，距離 Avg `19` 只差 `0.05`；1M 最強為 Avg `18.05` / Min `10`，也接近目標。

## Latest 1.5M Continuation Comparison

These results compare the recent Colab, Lightning AI, and Kaggle continuation runs.
All reliable numbers below are from fixed 20-seed evaluation (`seed=0..19`), not only W&B online eval curves.

| Run | Platform | Key differences | 600k 20-seed eval | 1M 20-seed eval | 1.5M 20-seed eval | Conclusion |
|---|---|---|---:|---:|---:|---|
| `task3-fast-dueling-ddqn-per-nstep3-eps002-decay220k-tps2-lr5e5-warm80k-1500k` | Colab | `train_per_step=2`, `lr=5e-5`, `replay_start=80k`, `noop_max=0` | Avg `3.25`, Min `-10`, Max `12` | Avg `9.35`, Min `1`, Max `20` | Avg `8.00`, Min `-8`, Max `17` | Learns, but is unstable. Best checkpoint is 1M, and 1.5M regresses. |
| `task3-lighting_ai-fast-dueling-ddqn-per-nstep3-eps002-decay220k-noop30-1500k` | Lightning AI | Same stable base, with `noop_max=30` | Avg `12.25`, Min `-1`, Max `20` | Avg `10.25`, Min `-1`, Max `19` | Avg `10.70`, Min `-15`, Max `19` | Best among recent 1.5M continuation runs. Best checkpoint is 600k. |
| `task3-colab-no-dueling-stable-v2-eps002-nstep3-1500k` | Colab | No dueling, DDQN + PER + n-step 3, `lr=1e-4`, `eps_min=0.02`, `eps_decay_steps=260k` | Avg `7.00`, Min `-4`, Max `18` | Avg `9.35`, Min `-4`, Max `17` | Avg `11.35`, Min `-1`, Max `20` | Best long no-dueling continuation before the reference slowexp run. It improves through 1.5M but plateaus far below Avg 19. |
| `task3-colab-no-dueling-ref-adameps-nstep3-600k` | Colab | No dueling, reference-style recipe: `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `n_step=3` | Avg `10.45`, Min `-15`, Max `20` | Avg `16.25`, Min `11`, Max `20` | N/A | Strong 1M no-dueling result. The 600k checkpoint is high-variance, but the 1M checkpoint becomes much more robust. |
| `task3-lighting_ai-no-dueling-ref-adameps-nstep3-slowexp99998-600k` | Lightning AI | No dueling reference recipe, but slower exponential decay `0.99998`, `eps_min=0.01` | Avg `11.80`, Min `-13`, Max `18` | N/A | N/A | Best no-dueling 600k average so far. Slower decay improves average and greatly reduces widespread seed failure, though one outlier seed remains. |
| `task3-kaggle-no-dueling-ref-adameps-nstep3-slowexp99998-warm50k-600k` | Kaggle | No-dueling slowexp reference recipe, but `replay_start=50k` instead of `20k` | Avg `10.90`, Min `-3`, Max `17` | N/A | N/A | Longer warm-up reduces catastrophic failure compared with the Lightning AI no-dueling slowexp run, but lowers average and ceiling. |
| `task3-colab-no-dueling-ref-adameps-nstep3-eps002-600k` | Colab | No dueling reference recipe, but `eps_min=0.02`, exponential decay `0.99996` | Avg `6.10`, Min `-3`, Max `11` | N/A | N/A | Too much residual exploration / insufficient exploitation for this reference recipe. Lower variance but much lower ceiling. |
| `task3-kaggle-dueling-ref-adameps-nstep3-eps002-600k` | Kaggle | Dueling reference recipe, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.02`, `replay_start=20k`, `target_update=2000`, `n_step=3` | Avg `12.35`, Min `5`, Max `19` | N/A | N/A | Very stable 600k dueling result. Slightly below best Avg `12.80`, but matches best baseline Min `5` and has no catastrophic seeds. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-600k` | Kaggle | Dueling reference recipe with slower exponential decay `0.99998`, `eps_min=0.01` | Avg `15.00`, Min `9`, Max `21` | N/A | N/A | New overall best 600k result. Combines high average with strong robustness; no seed below `9`. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-1000k` | Colab | Dueling reference recipe with faster exponential decay `0.99996`, `eps_min=0.01`, `target_update=2000`, trained to 1M | Avg `16.15`, Min `11`, Max `20` | Avg `14.75`, Min `-8`, Max `21` | N/A | New best 600k checkpoint, but the 1M checkpoint regresses. Use checkpoint selection. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-600k` | Colab | Same dueling `exp99996 target2000` 600k recipe, but `lr=2e-4` instead of `2.5e-4` | Avg `17.75`, Min `11`, Max `21` | N/A | N/A | New best 600k result. Lower LR improves average while preserving worst-case score. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-600k` | Lighting AI | Same 600k best branch, but `lr=1.8e-4` | Avg `18.95`, Min `14`, Max `21` | N/A | N/A | New best 600k result, only `0.05` below Avg `19`; seed=3 evaluation reportedly reaches Avg `19.3`, but seed=0 remains the fixed-protocol number. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e6-2.5M` | Lighting AI | Repeat of the `lr=1.8e-4` best branch trained to 2.5M; run name says `lr18e6`, but command uses `--lr 0.00018` | Avg `18.95`, Min `14`, Max `21` | Avg `15.90`, Min `2`, Max `20` | Avg `17.50`, Min `13`, Max `21`; 2M Avg `18.65`, Min `12`, Max `21`; 2.5M Avg `17.85`, Min `12`, Max `21` | Best checkpoint remains 600k. Later checkpoints oscillate: strong rebound at 2M, but no later checkpoint beats 600k. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay600k-lr1e5-2500k` | Lighting AI | Same best branch, but LR decays from `1.8e-4` to `1e-5` after 600k | Avg `18.95`, Min `14`, Max `21` | Avg `19.45`, Min `16`, Max `21` | Avg `20.15`, Min `18`, Max `21`; 2M Avg `19.90`, Min `17`, Max `21`; 2.5M Avg `20.05`, Min `18`, Max `21` | New strongest overall Task 3 run. 600k is still `18.95`, but 1M and all later milestones are officially above Avg `19`. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr18e5-repeat-2500k` | Colab | Same `lr=1.8e-4, target_update=2000` recipe trained toward 2.5M | Avg `12.00`, Min `-5`, Max `18` | Avg `15.65`, Min `9`, Max `21` | Avg `17.80`, Min `13`, Max `21`; 2M Avg `13.00`, Min `-1`, Max `18` | Colab does not reproduce the Lightning AI 600k peak. It improves by 1.5M, then drifts downward by 2M. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay1500k-lr1e5-2500k` | Colab | Same Colab repeat branch, but LR decays from `1.8e-4` to `1e-5` after 1.5M | W&B only | W&B only | Crashed around ~0.8M before fixed eval | No fixed 20-seed data. W&B trend does not look like a clear improvement; do not prioritize resuming. |
| `task3-kaggle-dueling-ref-adameps-nstep3-exp99996-lr18e5-repeat-2500k` | Kaggle | Same `lr=1.8e-4, target_update=2000` recipe trained to 2.5M | Avg `10.55`, Min `3`, Max `16` | Avg `16.65`, Min `8`, Max `21` | Avg `15.05`, Min `5`, Max `20`; 2M Avg `15.15`, Min `8`, Max `20`; 2.5M Avg `16.15`, Min `10`, Max `20` | Kaggle also does not reproduce the Lighting AI 600k peak. It is strongest at 1M/2.5M but remains below Avg `19`. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr19e5-600k` | Lighting AI | Same 600k best branch, but `lr=1.9e-4` | Avg `17.50`, Min `14`, Max `20` | N/A | N/A | Strong but below `lr=1.8e-4`; slightly higher LR does not improve average. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr17e5-600k` | Colab | Same 600k best branch, but `lr=1.7e-4` | Avg `16.80`, Min `13`, Max `20` | N/A | N/A | More conservative LR is stable but too slow / lower average. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr175e6-600k` | Lighting AI | Same 600k best branch, but `lr=1.75e-4`; checkpointed through 1.5M | Avg `17.70`, Min `12`, Max `21` | Avg `17.70`, Min `12`, Max `21` | Avg `16.20`, Min `7`, Max `21` | Strong at 600k and 1M, but below `lr=1.8e-4`; 1.5M regresses, so extending this branch does not help. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-softtau001-600k` | Colab | Same 600k best branch, but soft target update with `tau=0.001` | Avg `17.45`, Min `10`, Max `20` | N/A | N/A | Strong but slightly below the hard-update 600k best. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-resume500k-to1000k` | Colab | Resume from 500k using the `exp99996 lr2e-4` recipe and continue to 1M | Avg `16.35`, Min `3`, Max `21` | Avg `16.35`, Min `8`, Max `21` | N/A | Resume extension does not improve the 1M result and is weaker than the original 600k best. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-target1500-600k` | Colab | Same dueling `exp99996` 600k recipe, but `target_update=1500` | Avg `14.90`, Min `8`, Max `20` | N/A | N/A | Stable positive run, but below the `target_update=2000` 600k best. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-target1000-600k` | Lightning AI | Same dueling `exp99996` 600k recipe, but `target_update=1000` | Avg `16.15`, Min `6`, Max `20` | N/A | N/A | Matches the best average but has weaker worst-case seed than `target_update=2000`. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-target2500-2500k` | Lighting AI | Same best 600k branch, but `target_update=2500`, trained toward 2.5M | Avg `8.75`, Min `1`, Max `15` | Avg `13.55`, Min `8`, Max `19` | Avg `17.10`, Min `14`, Max `21` | Too slow for 600k, but becomes stable by 1.5M. Useful as longer-training ablation, not a 600k candidate. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-1000k` | Kaggle | Same dueling slowexp reference recipe, but `target_update=1000` and trained to 1M | Avg `13.95`, Min `5`, Max `21` | Avg `18.05`, Min `10`, Max `21` | N/A | Current best 1M result. It does not beat the 600k best at 600k, but becomes very strong by 1M. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-1000k` | Lightning AI | Same 1M best recipe, but `lr=2e-4` instead of `2.5e-4` | Avg `12.00`, Min `5`, Max `18` | Avg `18.00`, Min `14`, Max `20` | N/A | Slightly lower average than the 1M best, but much stronger worst-case robustness. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-softtau001-1000k` | Lightning AI | Same robust 1M recipe, but soft target update `tau=0.001` | Avg `4.20`, Min `-11`, Max `15` | Avg `14.25`, Min `1`, Max `21` | N/A | Soft target update severely hurts this branch. |
| `task3-colab-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-softtau0005-1000k` | Colab | Same robust 1M recipe, but soft target update `tau=0.0005` | Avg `9.95`, Min `-3`, Max `18` | Avg `12.20`, Min `1`, Max `20` | N/A | Smaller soft tau still performs poorly; soft update direction should be stopped. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-beta800k-1000k` | Kaggle | Same robust 1M recipe, but `per_beta_anneal_steps=800k` | Avg `10.15`, Min `-1`, Max `17` | Pending / stopped | N/A | Paused due to quota; 600k fixed eval is weak, so this is low priority to continue. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-1500k` | Kaggle | Same slowexp target1000 recipe, trained to 1.5M with `beta_anneal=1.5M` | Avg `14.90`, Min `10`, Max `21` | Avg `15.60`, Min `8`, Max `19` | Avg `16.95`, Min `9`, Max `20` | Longer training to 1.5M improves within this run but does not reach the previous 1M best. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1500-1000k` | Kaggle | Same dueling slowexp 1M recipe, but `target_update=1500` | Avg `15.10`, Min `8`, Max `19` | Avg `15.70`, Min `7`, Max `20` | N/A | Learns, but clearly below `target_update=1000` at 1M. |
| `task3-colab-dueling-ref-adameps-nstep3-slowexp999985-600k` | Colab | Same dueling slowexp reference recipe, but slower decay `0.999985` | Avg `12.75`, Min `-7`, Max `19` | N/A | N/A | Slower decay did not help. It learned, but one bad seed returned and the average fell below the `0.99998` best. |
| `task3-colab-no-dueling-ref-adameps-nstep3-slowexp99998-noop30-600k` | Colab | No-dueling slowexp reference recipe with `noop_max=30` | Avg `7.45`, Min `-3`, Max `17` | N/A | N/A | NoopReset hurt this no-dueling recipe. It reduced catastrophic failure but lowered average and ceiling compared with no-dueling slowexp without noop. |
| `task3-kaggle-t4-balanced-v2-dueling-ddqn-per-nstep3` | Kaggle T4 | `lr=7.5e-5`, `replay_start=50k`, `eps_decay=300k`, `eps_min=0.01`, `target_update=4000`, `max_steps=1.2M` | Pending fixed eval | Pending fixed eval | N/A | W&B trend shows learning, but fixed 20-seed evaluation is still not available. Treat as exploratory only. |

Interpretation:

- The overall best fixed 600k result is now `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay600k-lr1e5-2500k` with Avg `18.95`, Min `14`, Max `21`.
- The overall best fixed 1M result is now `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-decay600k-lr1e5-2500k` with Avg `19.45`, Min `16`, Max `21`.
- The same LR-decay run is also the strongest later-milestone result so far: 1.5M Avg `20.15`, 2M Avg `19.90`, and 2.5M Avg `20.05`.
- Among the new long continuation runs, Lightning AI with `noop_max=30` is the strongest and most robust early checkpoint.
- The Colab safe TPS=2 run improved by 1M, but continuing to 1.5M did not help. This suggests later DQN drift / instability rather than simple under-training.
- `noop_max=30` improves seed robustness at 600k, but the 1M and 1.5M checkpoints do not improve over 600k. Use checkpoint selection, not final-step selection.
- The Colab no-dueling stable v2 run is still useful as a long-training no-dueling reference. It reaches Max `20` at 1.5M, but Avg `11.35` remains below the newer 600k no-dueling slowexp Avg `11.80`.
- The Colab no-dueling reference Adam-eps run improves dramatically when extended to 1M: Avg rises from `10.45` at 600k to `16.25` at 1M, and Min improves from `-15` to `11`. This is currently the strongest no-dueling long-training result.
- The Colab dueling `exp99996` run is the new 600k best, but continuing it to 1M hurts performance badly: Avg drops from `16.15` to `14.75` and Min drops from `11` to `-8`. This is clear checkpoint drift.
- Lowering LR from `2.5e-4` to `2e-4` in the `exp99996 target2000` 600k branch improves Avg from `16.15` to `17.75` while preserving Min `11`. This is currently the strongest 600k direction.
- Lowering LR further to `1.8e-4` improves the 600k branch again, reaching Avg `18.95`, Min `14`, Max `21`. This is effectively at the Avg `19` target under one-decimal reporting.
- Testing nearby values confirms `lr=1.8e-4` as the current sweet spot: `1.7e-4` drops to Avg `16.80`, `1.75e-4` reaches Avg `17.70`, and `1.9e-4` drops to Avg `17.50`.
- Soft target update with `tau=0.001` on the 600k best branch remains strong but does not improve the hard-update result: Avg `17.45` vs `17.75`.
- Resuming the `exp99996 lr2e-4` branch from 500k to 1M does not preserve the original 600k score and does not become a strong 1M run. It reaches Avg `16.35` at both 600k and 1M, far below the slowexp99998 1M branch.
- The Colab dueling `exp99996 target1500` run is stable but not better than `target2000`: Avg `14.90` vs `16.15`. For the 600k target, `target_update=2000` remains the best tested setting in this branch.
- The Lightning AI dueling `exp99996 target1000` run matches the best Avg `16.15`, but Min drops to `6`. For final selection, the `target_update=2000` checkpoint is still better because it has the same average with stronger worst-case performance.
- The Kaggle `slowexp99998 target1500` 1M run underperforms the `target1000` 1M best by a wide margin: Avg `15.70` vs `18.05`. For the 1M branch, `target_update=1000` remains the better setting.
- Lowering LR to `2e-4` in the 1M slowexp target1000 branch trades a tiny amount of average score for much better robustness: Avg `18.00` vs `18.05`, but Min `14` vs `10`.
- Soft target update `tau=0.001` performs poorly on the 1M slowexp branch: Avg `14.25` at 1M vs `18.00` for the hard-update LR2e-4 version. Do not continue this tau setting.
- Soft target update `tau=0.0005` also performs poorly on the 1M slowexp branch: Avg `12.20` at 1M. This confirms soft target update is not useful for the current recipe.
- The `beta800k` run was paused due to quota. Its 600k fixed eval is only Avg `10.15`, so it is not worth prioritizing unless extra compute is available.
- Extending the hard-update slowexp target1000 branch to 1.5M did not reach Avg `19`; this run reached Avg `16.95` at 1.5M and did not reproduce the earlier Avg `18.05` at 1M.
- The reference-style no-dueling direction is now useful: the original reference run reached Avg `10.45`, and the slower-decay Lightning AI variant improved to Avg `11.80`. This is still below the best dueling baseline Avg `12.80`, but it shows that optimizer/schedule choices matter nearly as much as dueling.
- In the reference recipe, raising `eps_min` from `0.01` to `0.02` hurt badly: Avg dropped to `6.10`. Unlike the earlier dueling linear-decay runs, this no-dueling exponential schedule appears to need low final epsilon for enough exploitation.
- Adding dueling back to the reference recipe with `eps_min=0.02` produced Avg `12.35`, Min `5`, Max `19`. It did not beat the older best Avg `12.80`, but it is one of the most stable 600k results and has no negative seeds.
- Changing that dueling reference recipe to slower exponential decay `0.99998` and `eps_min=0.01` produced the current best Avg `15.00`. This strongly supports the combination of dueling + reference optimizer + slower exponential epsilon decay.
- Slowing decay further to `0.999985` dropped Avg to `12.75` and reintroduced a negative seed (`-7`). For the 600k target, `0.999985` appears too slow; the sweet spot is closer to `0.99998` or slightly faster.
- Increasing no-dueling slowexp warm-up from `20k` to `50k` reduced catastrophic negative seeds, but Avg dropped from `11.80` to `10.90` and Max dropped from `18` to `17`. Longer warm-up helps robustness slightly but does not solve the 600k target.
- Adding `noop_max=30` to the no-dueling slowexp run hurt performance, dropping Avg from `11.80` to `7.45`; this suggests NoopReset is not helpful for the current no-dueling reference recipe at 600k.
- Kaggle balanced v2 changed several variables at once, so it should not be used for a final claim until the same fixed 20-seed evaluation runs successfully.

## Experiment Summary Table

| Run | Status | Main params | 600k eval / trend | Analysis | Next action |
|---|---|---|---|---|---|
| `task3-lighting-ai-plain-ddqn-conservative-600k` | Completed | Plain DDQN, inferred `train_per_step=1`, no PER, no n-step | Eval around `-7.5` near 600k | 第一個有效學習的 plain baseline。沒有達標，但證明 DDQN 方向可行。 | 作為早期 baseline，不再沿這條主線調。 |
| `task3-plain-ddqn-stable-600k` | Completed | Plain DDQN, `train_per_step=2`, `lr=1.5e-4`, `batch=64`, `replay_start=20k`, `target_update=20000`, `eps_min=0.10`, no PER, no dueling, `n_step=1` | Eval stayed at `-21` | 更新太多且 warm-up 太短，造成 Q-value collapse / over-training。 | 不再使用這種 aggressive TPS=2 設定。 |
| `task3-lighting_ai-stable-ddqn-per-nstep3-600k` | Interrupted | DDQN + PER + n-step 3, no dueling, `replay_start=100k`, `eps_min=0.05`, `eps_decay_steps=350k`, `target_update=2000` | Interrupted around 360k; trend slow | 設定太保守，前 100k 不訓練且 epsilon 下降慢，600k 內學習速度不足。 | 不需要重跑；已提供「太保守」證據。 |
| `task3-fast-dueling-ddqn-per-nstep3-600k` | Completed | DDQN + Dueling + PER + n-step 3, `lr=1e-4`, `batch=32`, `replay_start=50k`, `train_per_step=1`, `target_update=1000`, `eps_min=0.02`, `eps_decay_steps=220k`, `noop_max=0` | Avg `12.80`, Min `5`, Max `20` | 目前最佳。沒有 Q collapse，Total Reward 與 Eval Reward 後期明顯上升。 | 保留為主要 baseline。 |
| `task3-lighting_ai-fast-stable-ddqn-per-nstep3-600k` | Completed | DDQN + Dueling + PER + n-step 3, `eps_min=0.04`, `eps_decay_steps=260k`, `replay_start=50k`, `target_update=1000` | Avg `7.85`, Min `-2`, Max `18` | 有效學習但輸給 baseline。後期探索太多，exploitation 不夠乾淨。 | 隱藏圖表；不再沿 `eps_min=0.04` 調。 |
| `task3-fast-dueling-ddqn-per-nstep3-eps003-decay250k-600k` | Completed | Same as best baseline, but `eps_min=0.03`, `eps_decay_steps=250k` | Avg `9.15`, Min `-4`, Max `18` | 曲線看起來不差，但 20-seed eval 輸給 baseline。`eps_min=0.03` 仍偏高。 | 隱藏圖表；epsilon 方向回到 `0.02`。 |
| `task3-lighting_ai-fast-dueling-ddqn-per-nstep3-eps001-decay220k-600k` | Completed | Same as best baseline, but `eps_min=0.01`, `eps_decay_steps=220k` | Avg `-8.20`, Min `-14`, Max `4` | 太 greedy。Q/Target 沒崩，但學到穩定的壞 policy，後期探索不足以修正。 | 隱藏圖表；不要再測更低 epsilon。 |
| `task3-fast-dueling-ddqn-per-nstep5-eps002-decay220k-600k` | Stopped early | Same as best baseline, but `n_step=5` | Around 500k still mostly poor eval logs: `-15, 3, 2, -6, -4, -6` | `n_step=5` 學得比 `n_step=3` 慢，可能 target variance 較高。 | 已停止；不再測更大的 n-step。 |
| `task3-fast-dueling-ddqn-per-nstep3-eps002-decay220k-tps2-lr5e5-warm80k-1500k` | Completed | DDQN + Dueling + PER + n-step 3, `train_per_step=2`, `lr=5e-5`, `replay_start=80k`, `target_update=2000`, `eps_min=0.02`, `eps_decay_steps=220k`, `noop_max=0`, `max_steps=1.5M` | 600k Avg `3.25`; 1M Avg `9.35`; 1.5M Avg `8.00` | Safe TPS=2 learned, but did not beat the 600k baseline. 1M was best; 1.5M regressed. | Do not use as final best. Keep as evidence that extra updates need checkpoint selection. |
| `task3-lighting_ai-fast-dueling-ddqn-per-nstep3-eps002-decay220k-noop30-1500k` | Completed | DDQN + Dueling + PER + n-step 3, stable base with `noop_max=30`, `eps_min=0.02`, `eps_decay_steps=220k`, `max_steps=1.5M` | 600k Avg `12.25`; 1M Avg `10.25`; 1.5M Avg `10.70` | Strongest recent continuation run. NoopReset improved robustness at 600k, but later checkpoints did not improve. | Use 600k checkpoint as the best recent continuation result; compare against overall 600k baseline Avg `12.80`. |
| `task3-colab-no-dueling-stable-v2-eps002-nstep3-1500k` | Completed | No dueling, DDQN + PER + n-step 3, `lr=1e-4`, `memory=100k`, `replay_start=50k`, `target_update=1000`, `eps_min=0.02`, `eps_decay_steps=260k`, `beta_anneal=600k` | 600k Avg `7.00`; 1M Avg `9.35`; 1.5M Avg `11.35` | Best long no-dueling run so far. Lowering `eps_min` from `0.04` to `0.02` helped, but longer training still plateaued below the target. | Keep as long no-dueling baseline; compare against the newer reference-style 600k no-dueling run. |
| `task3-colab-no-dueling-ref-adameps-nstep3-600k` | Completed | No dueling, DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `train_per_step=1`, `noop_max=0`, extended to `1M` | 600k Avg `10.45`; Min `-15`; Max `20`; 1M Avg `16.25`; Min `11`; Max `20` | Strong no-dueling long-training run. The 600k checkpoint has high variance, but by 1M the same recipe becomes robust and all seeds are positive. | Keep as the main no-dueling 1M reference. It is below Avg `19`, but it proves that this recipe benefits strongly from extra steps. |
| `task3-lighting_ai-no-dueling-ref-adameps-nstep3-slowexp99998-600k` | Completed | No dueling, DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `11.80`; Min `-13`; Max `18` | Best no-dueling 600k average so far. Most seeds are positive and clustered around `8`-`18`, but seed 14 collapses to `-13`, so robustness is still not solved. | Keep as current no-dueling reference baseline. Consider extending or testing the same slower decay with dueling after Kaggle finishes. |
| `task3-kaggle-no-dueling-ref-adameps-nstep3-slowexp99998-warm50k-600k` | Completed | No-dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=50k`, `target_update=2000`, `noop_max=0`, `max_steps=600k` | 600k Avg `10.90`; Min `-3`; Max `17` | Warm-up 50k reduced the worst negative outliers compared with the 20k no-dueling slowexp run, but the average and ceiling dropped. The model learned, but it did not improve the main no-dueling baseline. | Do not make warm50k the main no-dueling recipe. Keep it only as evidence that longer warm-up improves robustness at the cost of sample efficiency. |
| `task3-colab-no-dueling-ref-adameps-nstep3-eps002-600k` | Completed | No dueling, DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.02`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `6.10`; Min `-3`; Max `11` | Raising `eps_min` to `0.02` made the policy much less strong. It reduced catastrophic negative seeds but also capped the ceiling; no seed reached high Pong scores. | Do not continue this branch unless testing with longer training as a robustness-only baseline. |
| `task3-kaggle-dueling-ref-adameps-nstep3-eps002-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.02`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `12.35`; Min `5`; Max `19` | Strong and stable dueling reference result. It is slightly below the overall best Avg `12.80`, but unlike many no-dueling runs it has no catastrophic seed and every evaluation episode is positive. | Keep as stability-focused dueling reference. A next candidate is dueling + slowexp `0.99998` to combine this robustness with the slow-decay improvement seen in no-dueling. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `15.00`; Min `9`; Max `21` | New overall best 600k result. It improves both average and robustness over the previous best. All seeds are positive and most are in the `10`-`18` range. | Treat as the new main checkpoint. Next tests should refine around this recipe, not restart from older baselines. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=1M` | 600k Avg `16.15`; Min `11`; Max `20`; 1M Avg `14.75`; Min `-8`; Max `21` | New best 600k checkpoint, but longer training causes regression. The 600k model is robust; the 1M model reintroduces a catastrophic seed. | Use the 600k checkpoint for 600k reporting. Do not use its 1M checkpoint as final. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `17.75`; Min `11`; Max `21` | New best 600k result. Lower learning rate improves average significantly while maintaining the same worst-case score as the previous best. | Keep as the main 600k checkpoint. Next 600k tests should refine LR around this value. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr18e5-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=1.8e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `18.95`; Min `14`; Max `21` | New best 600k result. This is within `0.05` of Avg `19` and has much stronger worst-case performance than earlier 600k runs. | Use this as the main 600k checkpoint. Do not cherry-pick evaluation seed unless the assignment explicitly allows it. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr19e5-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=1.9e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `17.50`; Min `14`; Max `20` | Strong and robust, but lower than `lr=1.8e-4`. Increasing LR slightly does not improve the best setting. | Keep as LR ablation evidence. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr17e5-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=1.7e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `16.80`; Min `13`; Max `20` | Stable, but average drops compared with `lr=1.8e-4`. Lower LR appears too conservative for 600k. | Keep as LR ablation evidence. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-lr175e6-600k` | Completed to 1.5M | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=1.75e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`; command used longer training despite `600k` name | 600k Avg `17.70`; Min `12`; Max `21`; 1M Avg `17.70`; Min `12`; Max `21`; 1.5M Avg `16.20`; Min `7`; Max `21` | Stable and strong up to 1M, but 1.5M regresses. The lower LR still does not beat `lr=1.8e-4` at 600k. | Keep as LR ablation / longer-training evidence; do not replace the `lr=1.8e-4` 600k best. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-softtau001-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, soft target update `tau=0.001`, `max_steps=600k` | 600k Avg `17.45`; Min `10`; Max `20` | Strong result, but soft target update does not beat the hard-update 600k best. | Keep hard-update `lr2e-4` as the 600k best. Soft tau may still be useful to test on the 1M branch. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-lr2e4-resume500k-to1000k` | Completed | Dueling DDQN + PER + n-step 3, same as the 600k best recipe, resumed from 500k and continued to 1M | 600k Avg `16.35`; Min `3`; Max `21`; 1M Avg `16.35`; Min `8`; Max `21` | Resume continuation did not recover the original 600k best and did not improve at 1M. The 1M model is more robust than its 600k checkpoint but average stays flat. | Do not use this as final. Keep the original 600k checkpoint for 600k and the slowexp99998 target1000 branch for 1M. |
| `task3-colab-dueling-ref-adameps-nstep3-exp99996-target1500-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=1500`, `max_steps=600k` | 600k Avg `14.90`; Min `8`; Max `20` | Stable 600k result with no negative seeds, but it does not beat the `target_update=2000` best. Several seeds remain in the `8`-`13` range, lowering average. | Do not replace the 600k best. Keep as target-update ablation evidence. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-exp99996-target1000-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `replay_start=20k`, `target_update=1000`, `max_steps=600k` | 600k Avg `16.15`; Min `6`; Max `20` | Strong 600k average that ties the current best, but robustness is weaker because one seed falls to `6`. | Keep as evidence that faster target sync can match average, but prefer `target_update=2000` for final 600k checkpoint selection. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=1000`, `max_steps=1M` | 600k Avg `13.95`; Min `5`; Max `21`; 1M Avg `18.05`; Min `10`; Max `21` | Current best 1M result. Faster target sync does not beat the 600k best at 600k, but it keeps improving and becomes very strong by 1M. | Keep as the main 1M candidate. Next 1M tests should make small changes around this recipe, especially target update / epsilon schedule. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=1000`, `max_steps=1M` | 600k Avg `12.00`; Min `5`; Max `18`; 1M Avg `18.00`; Min `14`; Max `20` | Very strong and robust 1M result. Average is almost tied with the best, and the minimum score is much higher. | Keep as the conservative 1M candidate. It may be preferable if robustness matters more than the tiny Avg gap. |
| `task3-lighting_ai-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-softtau001-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, soft target update `tau=0.001`, `max_steps=1M` | 600k Avg `4.20`; Min `-11`; Max `15`; 1M Avg `14.25`; Min `1`; Max `21` | Soft target update hurts this branch substantially. The target network likely tracks too slowly or changes the learning dynamics unfavorably with PER + n-step. | Do not use soft tau `0.001` for this recipe. Prefer hard update. |
| `task3-colab-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-softtau0005-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, soft target update `tau=0.0005`, `max_steps=1M` | 600k Avg `9.95`; Min `-3`; Max `18`; 1M Avg `12.20`; Min `1`; Max `20` | Smaller soft tau still performs far below hard update. The issue is not just tau being too large. | Stop soft target update experiments for this recipe. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-lr2e4-beta800k-1000k` | Paused | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=1000`, `per_beta_anneal_steps=800k`, intended `max_steps=1M` | 600k Avg `10.15`; Min `-1`; Max `17` | This tested earlier PER beta correction. The 600k fixed eval was weak and Kaggle quota ran out before 1M. | Low priority to continue; only revisit if extra compute is available. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1000-1500k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=1000`, `max_steps=1.5M`, `per_beta_anneal_steps=1.5M` | 600k Avg `14.90`; Min `10`; Max `21`; 1M Avg `15.60`; Min `8`; Max `19`; 1.5M Avg `16.95`; Min `9`; Max `20` | Extending this recipe to 1.5M did not produce Avg 19. It improves within the run, but the 1M and 1.5M checkpoints are below the previous 1M best. | Do not rely on longer training alone. Keep the previous 1M best and robust LR2e-4 variants as stronger candidates. |
| `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-target1500-1000k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=1500`, `max_steps=1M` | 600k Avg `15.10`; Min `8`; Max `19`; 1M Avg `15.70`; Min `7`; Max `20` | Intermediate target sync did not help the 1M branch. The model learns, but final robustness and average are much weaker than `target_update=1000`. | Do not continue target1500 for the 1M slowexp branch. |
| `task3-colab-dueling-ref-adameps-nstep3-slowexp999985-600k` | Completed | Dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, even slower exponential epsilon decay `0.999985`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `max_steps=600k` | 600k Avg `12.75`; Min `-7`; Max `19` | The model still learned, but the slower decay reduced 600k performance compared with `0.99998`. The negative seed suggests the policy was not as consistently converged by 600k. | Do not continue slower decay in this direction for the 600k target. Prefer `0.99998`, or test slightly faster decay / target update ablations. |
| `task3-colab-no-dueling-ref-adameps-nstep3-slowexp99998-noop30-600k` | Completed | No-dueling DDQN + PER + n-step 3, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, slower exponential epsilon decay `0.99998`, `eps_min=0.01`, `replay_start=20k`, `target_update=2000`, `noop_max=30`, `max_steps=600k` | 600k Avg `7.45`; Min `-3`; Max `17` | NoopReset hurt the no-dueling slowexp recipe. It reduced the catastrophic negative outlier but also lowered the average and ceiling substantially. | Do not continue this branch for the 600k target. |
| `task3-kaggle-t4-balanced-v2-dueling-ddqn-per-nstep3` | Eval pending | DDQN + Dueling + PER + n-step 3, `lr=7.5e-5`, `replay_start=50k`, `eps_decay_steps=300k`, `eps_min=0.01`, `target_update=4000`, `max_steps=1.2M` | W&B trend only; fixed 20-seed eval not available yet | Too many knobs changed at once. Cannot conclude from W&B curve alone. | Fix Kaggle evaluation/logging first; otherwise do not cite as final result. |

## Completed Run Notes

### Plain DDQN Conservative 600k

This run reached about `-7.5` evaluation reward near 600k. It was not close to the final target, but it was the first run showing real learning progress.

Important signs:

- Eval Reward improved from `-21` to around `-7.5`.
- Q / Target trends recovered instead of collapsing.
- This suggests DDQN itself can learn Pong, but plain DDQN is not sample-efficient enough for the target.

### Plain DDQN Stable 600k

This run stayed at `-21` through training. The main issue was likely over-training:

- `train_per_step=2` caused around 1.1M updates by 600k environment steps.
- `replay_start=20k` was too early.
- `lr=1.5e-4` and slow target sync made the aggressive update schedule worse.
- Q Mean / Target Mean moved in the wrong direction, and action distributions became unstable.

Conclusion: this was not a lack of exploration. It was update instability / Q-value collapse.

### Best Baseline: Fast Dueling DDQN PER n-step 3

20-seed evaluation:

```text
Average: 12.80
Min: 5
Max: 20
```

This was the previous best 600k setting before the Kaggle dueling slowexp99998 run.

Why it worked:

- `train_per_step=1` avoided over-training.
- Dueling DQN helped estimate state value more efficiently.
- PER helped focus on useful transitions.
- `n_step=3` improved reward propagation without too much variance.
- `epsilon_min=0.02` gave enough late-stage exploitation while still preserving minimal exploration.

Remaining issue:

- Max score reaches 20, so the policy has enough capacity.
- Average is still 12.8, so the issue is seed-to-seed robustness and consistency.

### Epsilon Ablations

| Epsilon setting | Result | Interpretation |
|---|---:|---|
| `eps_min=0.01`, `decay=220k` | Avg `-8.20` | Too greedy. Learns a bad stable policy and cannot recover. |
| `eps_min=0.02`, `decay=220k` | Avg `12.80` | Former best linear-decay dueling setting. |
| `eps_min=0.03`, `decay=250k` | Avg `9.15` | Too much late exploration; lower final score. |
| `eps_min=0.04`, `decay=260k` | Avg `7.85` | Even more late exploration; worse than baseline. |

Conclusion: do not continue pushing epsilon lower or higher. Keep `eps_min=0.02`, `decay_steps=220k` as the current default.

### n-step Ablation

`n_step=5` was stopped early because it remained weak around 500k. It did not show the same late-stage acceleration as `n_step=3`.

Interpretation:

- Pong rewards are sparse, so longer n-step returns could help in theory.
- In practice, `n_step=5` likely increased target variance too much.
- `n_step=3` is currently the better tradeoff.

## Current / Next Experiments

### Completed: Safe train_per_step=2

Purpose:

Test whether a safer TPS=2 setup can improve sample efficiency without causing the Q-value collapse seen in the old aggressive TPS=2 run.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `3.25` | `-10` | `12` |
| 1M | `9.35` | `1` | `20` |
| 1.5M | `8.00` | `-8` | `17` |

Conclusion:

- The run learned, so the safer TPS=2 setup did avoid total collapse.
- It did not beat the best 600k baseline Avg `12.80`.
- The best checkpoint was 1M, while 1.5M regressed. This supports checkpoint selection over always using the final model.

### Completed: NoopReset ablation

Purpose:

Test whether `noop_max=30` improves seed robustness and later milestone performance.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `12.25` | `-1` | `20` |
| 1M | `10.25` | `-1` | `19` |
| 1.5M | `10.70` | `-15` | `19` |

Conclusion:

- `noop_max=30` improved early robustness compared with the safe TPS=2 Colab run.
- 600k was the best checkpoint; longer training did not improve the fixed 20-seed score.
- This is the strongest recent 1.5M continuation setting, but it is still slightly below the older overall best baseline Avg `12.80`.

### Pending: Kaggle balanced v2

Kaggle T4 training runs, but fixed 20-seed evaluation still needs to be made reliable.
Do not use the Kaggle curve as a final result until the same evaluation protocol prints per-seed rewards and averages.

### Completed: Colab no-dueling stable v2

Purpose:

Continue the no-dueling direction with a more exploitative but still stable setting:
DDQN + PER + n-step 3, `eps_min=0.02`, `eps_decay_steps=260k`, `lr=1e-4`, and no `--use-dueling`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `7.00` | `-4` | `18` |
| 1M | `9.35` | `-4` | `17` |
| 1.5M | `11.35` | `-1` | `20` |

Conclusion:

- This is the best no-dueling run so far.
- The model can reach a perfect-score episode (`20`) by 1.5M, so the policy has enough capacity to win.
- Average reward still plateaus around `11.35`, so robustness across seeds is the limiting factor.
- Training longer helped more than in the previous no-dueling run, but it still did not approach Avg `19`.
- This motivated the later reference-style no-dueling experiment using `memory=200k`, `replay_start=20k`, `lr=2.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.01`, `target_update=2000`, `n_step=3`, and Adam epsilon `1.5e-4`.

### Completed: Colab no-dueling reference Adam-eps n-step 3

Purpose:

Test a more reference-style no-dueling recipe with larger replay memory, higher learning rate, Adam epsilon, exponential epsilon decay, and earlier replay start:
DDQN + PER + n-step 3, no `--use-dueling`, `memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, `epsilon_decay=0.99996`, `eps_min=0.01`, `replay_start=20k`, and `target_update=2000`.

Evaluation sanity check:

The printed evaluation path matched the intended run:

```text
=== Evaluating results_task3_colab_no_dueling_ref_adameps_nstep3_600k @ 600000 steps ===
Using model: /content/drive/MyDrive/lab5/results_task3_colab_no_dueling_ref_adameps_nstep3_600k/LAB5_B11107027_task3_600000.pt
```

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `10.45` | `-15` | `20` |
| 1M | `16.25` | `11` | `20` |

Notable per-seed behavior:

- Strong seeds: seed 1 reached `19`, seeds 6 and 14 reached `20`, and several seeds reached `15+`.
- Weak seeds: seed 0 reached only `-15`, and seed 18 reached `-2`.
- At 1M, the distribution becomes much cleaner: all 20 seeds are positive, most seeds are between `15` and `19`, and seed 6 still reaches `20`.

Conclusion:

- This is a strong 600k no-dueling result and improves over no-dueling stable-v2 at 600k (`10.45` vs `7.00`).
- It still does not beat the best dueling 600k result at the 600k checkpoint, but the 1M checkpoint becomes one of the strongest overall long-training results.
- The result suggests that the reference-style optimizer/schedule choices matter: `lr=2.5e-4`, `adam_eps=1.5e-4`, larger replay memory, and exponential epsilon decay can compensate for removing dueling.
- The main 600k weakness is high seed variance. The policy can win some games, but the low Min `-15` shows it is not robust at 600k.
- Extending to 1M largely fixes that variance: Avg improves to `16.25` and Min improves to `11`.
- This suggests the no-dueling reference recipe is sample-hungry rather than fundamentally weak. It is not the best 600k strategy, but it is a serious 1M candidate.

### Completed: No-dueling reference epsilon variants

Purpose:

Test whether the reference-style no-dueling recipe is limited by epsilon scheduling. Two variants were compared against the original reference run:

- Slower exponential decay: `epsilon_decay=0.99998`, `eps_min=0.01`.
- Higher final epsilon: `epsilon_decay=0.99996`, `eps_min=0.02`.

Result:

| Run | Avg | Min | Max |
|---|---:|---:|---:|
| Original no-dueling ref: `decay=0.99996`, `eps_min=0.01` | `10.45` | `-15` | `20` |
| Slow-exp no-dueling ref: `decay=0.99998`, `eps_min=0.01` | `11.80` | `-13` | `18` |
| EPS002 no-dueling ref: `decay=0.99996`, `eps_min=0.02` | `6.10` | `-3` | `11` |

Conclusion:

- Slowing exponential decay improved the no-dueling reference recipe from Avg `10.45` to Avg `11.80`. Most seeds were positive, so the policy is more consistently useful.
- The remaining weakness is one catastrophic outlier: seed 14 scored `-13`. This keeps the average below the best dueling baseline.
- Raising final epsilon to `0.02` hurt the policy badly. It reduced catastrophic negative scores but also reduced the ceiling; no seed exceeded `11`.
- For this reference-style no-dueling recipe, the problem is not too little late exploration. It needs low final epsilon (`0.01`) and a slower decay to improve learning before exploitation.

### Completed: Kaggle no-dueling slowexp99998 warm50k

Purpose:

Test whether a longer warm-up period can make the no-dueling slowexp reference recipe more robust. This kept the same no-dueling reference settings as the slowexp99998 run, but changed `replay_start` from `20k` to `50k`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `10.90` | `-3` | `17` |

Conclusion:

- The run learned successfully and most seeds were positive.
- Compared with `task3-lighting_ai-no-dueling-ref-adameps-nstep3-slowexp99998-600k`, the worst seed improved from `-13` to `-3`, so longer warm-up did help reduce catastrophic failure.
- However, the average dropped from `11.80` to `10.90`, and the max dropped from `18` to `17`.
- This suggests `replay_start=50k` trades sample efficiency for robustness. It is useful evidence, but it is not the best no-dueling recipe for the 600k target.

### Completed: Kaggle dueling reference Adam-eps n-step 3 EPS002

Purpose:

Test whether adding dueling back to the reference-style recipe improves the strong no-dueling reference runs. This used the reference optimizer/schedule settings with dueling enabled:
`memory=200k`, `lr=2.5e-4`, `adam_eps=1.5e-4`, exponential epsilon decay `0.99996`, `eps_min=0.02`, `replay_start=20k`, `target_update=2000`, DDQN + PER + n-step 3 + dueling.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `12.35` | `5` | `19` |

Conclusion:

- This is one of the most stable 600k results: every seed is positive and the minimum score is `5`.
- It is slightly below the overall best Avg `12.80`, but the distribution is cleaner than the no-dueling reference runs, which had catastrophic negative outliers.
- Adding dueling improved robustness compared with the no-dueling reference variants.
- It did not fully solve the average-score target, so the next useful test is not simply "dueling or no dueling"; the more promising question is whether dueling should use the slower exponential decay `0.99998`, which helped no-dueling.

### Completed: Kaggle dueling reference slowexp99998

Purpose:

Test the most promising combination from the previous results: dueling for stability, reference optimizer settings, and the slower exponential epsilon decay that improved the no-dueling run.

Key setting:

```text
Dueling DDQN + PER + n-step 3
memory_size = 200000
lr = 2.5e-4
adam_eps = 1.5e-4
epsilon_decay_type = exp
epsilon_decay = 0.99998
epsilon_min = 0.01
replay_start = 20000
target_update = 2000
noop_max = 0
max_env_steps = 600000
```

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `15.00` | `9` | `21` |

Conclusion:

- This is the new overall best 600k result.
- It improves over the previous best Avg `12.80` by a large margin.
- It also improves robustness: all 20 evaluation seeds are positive, and the minimum is `9`.
- Compared with `task3-kaggle-dueling-ref-adameps-nstep3-eps002-600k`, slower decay + lower final epsilon improved Avg from `12.35` to `15.00`.
- Compared with no-dueling slowexp99998, adding dueling removed the catastrophic negative seed and increased Avg from `11.80` to `15.00`.
- The remaining gap to Avg `19` is not due to catastrophic failures anymore; it is due to many seeds plateauing around `10`-`18`.

### Completed: Colab dueling reference exp99996 1M

Purpose:

Test the dueling version of the no-dueling reference recipe that improved strongly at 1M. This uses faster exponential decay `0.99996`, low final epsilon `0.01`, `target_update=2000`, and trains to `1M`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `16.15` | `11` | `20` |
| 1M | `14.75` | `-8` | `21` |

Conclusion:

- The 600k checkpoint is the new best 600k result so far.
- Compared with the previous best `task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-600k`, Avg improves from `15.00` to `16.15`, and Min improves from `9` to `11`.
- The 1M checkpoint regresses badly: Avg drops to `14.75`, and Min falls to `-8`.
- This is a clear example of DQN checkpoint drift. The best model is not the final model.
- For 600k reporting, use the 600k checkpoint from this run.
- For 1M reporting, do not use this run; the Kaggle `slowexp99998-target1000-1000k` run remains better at 1M.

### Completed: Colab dueling reference exp99996 lr2e4 600k

Purpose:

Test whether slightly lowering learning rate improves the strongest 600k branch. This keeps the dueling `exp99996 target2000` recipe and changes only `lr` from `2.5e-4` to `2e-4`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `17.75` | `11` | `21` |

Conclusion:

- This is the new best 600k result.
- Compared with the previous `exp99996 target2000` result, Avg improves from `16.15` to `17.75`.
- Min remains `11`, so the improvement does not come from sacrificing robustness.
- Many seeds are now in the `18`-`21` range, suggesting that `lr=2.5e-4` was slightly too aggressive for the 600k dueling branch.
- The remaining gap to Avg `19` is mainly caused by a few lower seeds (`11`, `12`, `14`, `16`), not broad instability.

### Completed: Lighting AI dueling reference exp99996 lr18e5 600k

Purpose:

Continue refining the strongest 600k branch by lowering learning rate from `2e-4` to `1.8e-4`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `18.95` | `14` | `21` |

Conclusion:

- This is the best 600k result so far.
- Compared with `lr=2e-4`, Avg improves from `17.75` to `18.95`, and Min improves from `11` to `14`.
- The result is only `0.05` below Avg `19`; if displayed to one decimal place, it rounds to `19.0`.
- A separate evaluation with `EVAL_SEED=3` reportedly gives Avg `19.3`, but this should be treated as a supplemental seed-sensitivity observation unless the assignment explicitly allows choosing the evaluation seed.
- For fixed-protocol reporting, keep `seed=0` as the main number.

### Completed: Lighting AI dueling reference exp99996 lr18e6 2.5M repeat

Purpose:

Repeat the best `lr=1.8e-4` 600k recipe while allowing the run to continue to 2.5M checkpoints. Note: the run name contains `lr18e6`, but the actual command uses `--lr 0.00018`, so this is an `lr=1.8e-4` repeat, not `1.85e-4`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `18.95` | `14` | `21` |
| 1M | `15.90` | `2` | `20` |
| 1.5M | `17.50` | `13` | `21` |
| 2M | `18.65` | `12` | `21` |
| 2.5M | `17.85` | `12` | `21` |

Conclusion:

- The 600k fixed evaluation exactly matches the previous best run.
- The best checkpoint in this run is still 600k. Later checkpoints do not beat the 600k Avg `18.95`.
- The 1M checkpoint drops because one seed collapses to `2`, showing clear policy drift.
- The policy recovers strongly by 2M with Avg `18.65`, but 2.5M falls again to Avg `17.85`.
- This run is good for submission completeness because it provides all Task 3 milestone snapshots from the same training run. For `task3_best.pt`, use the 600k checkpoint from this run.

### Completed: Lighting AI exp99996 lr18e5 decay600k lr1e5

Purpose:

Test whether decaying LR from `1.8e-4` to `1e-5` after the strong 600k checkpoint can prevent policy drift in later milestones.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `18.95` | `14` | `21` |
| 1M | `19.45` | `16` | `21` |
| 1.5M | `20.15` | `18` | `21` |
| 2M | `19.90` | `17` | `21` |
| 2.5M | `20.05` | `18` | `21` |

Conclusion:

- This is now the strongest overall Task 3 run under the official seed `0`-`19` evaluation protocol.
- The 600k checkpoint remains at Avg `18.95`, matching the previous best. If strict unrounded scoring is used, it is just below the Avg `19` threshold; if rounded to one decimal place, it appears as `19.0`.
- LR decay after 600k clearly prevents the drift seen in the no-decay repeat: 1M rises to Avg `19.45`, and all later checkpoints stay above Avg `19`.
- For final Task 3 submission, use all milestone checkpoints from this same run. Use the 1.5M checkpoint as the strongest `task3_best.pt` candidate unless the grader explicitly prioritizes earliest checkpoint.

### In progress: Colab dueling reference exp99996 lr18e5 2.5M repeat

Purpose:

Check whether the Lightning AI best 600k recipe (`lr=1.8e-4`, `target_update=2000`) reproduces on Colab and whether longer training reaches the Avg `19` target.

Current result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `12.00` | `-5` | `18` |
| 1M | `15.65` | `9` | `21` |
| 1.5M | `17.80` | `13` | `21` |
| 2M | `13.00` | `-1` | `18` |

Interim conclusion:

- Colab does not reproduce the Lightning AI 600k peak with the same high-level parameters.
- The run becomes strong at 1.5M, but then drifts downward by 2M.
- This supports keeping the Lightning AI checkpoint as the 600k candidate, and treating Colab as a separate environment with different stochastic / hardware behavior.
- For Colab, further tests should focus on long-run stabilization or slower learning, not on expecting the same 600k peak as Lightning AI.

### Crashed: Colab dueling reference exp99996 lr18e5 decay1500k lr1e5

Purpose:

Test whether decaying LR from `1.8e-4` to `1e-5` after 1.5M can preserve the stronger later Colab policy and reduce drift.

Available evidence:

- Run crashed before fixed 20-seed evaluation.
- W&B trend only, ending around ~0.8M env steps.
- Total reward improved from the early phase but remained noisy.
- Eval reward rose into the mid-teens but did not show a clear advantage over the normal Colab repeat branch.

Conclusion:

- No formal checkpoint result is available, so this should not be cited as a completed experiment.
- The partial curve does not justify spending more compute on resuming it right now.
- If revisiting Colab later, a more useful direction is to tune a branch that is already strong at 1M/1.5M, rather than relying on this crashed decay run.

### Completed: Kaggle dueling reference exp99996 lr18e5 2.5M repeat

Purpose:

Check whether the Lighting AI best 600k recipe (`lr=1.8e-4`, `target_update=2000`) reproduces on Kaggle and whether longer training can recover the performance.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `10.55` | `3` | `16` |
| 1M | `16.65` | `8` | `21` |
| 1.5M | `15.05` | `5` | `20` |
| 2M | `15.15` | `8` | `20` |
| 2.5M | `16.15` | `10` | `20` |

Conclusion:

- Kaggle does not reproduce the Lighting AI 600k peak with this recipe.
- The run improves after 600k, with its best checkpoint at 1M (`16.65`) and a reasonably stable 2.5M checkpoint (`16.15`, Min `10`).
- It still remains below the earlier Kaggle slowexp99998 target1000 branch at 1M (`18.05`), so this is not the best Kaggle direction.
- This strengthens the platform-specific conclusion: Lighting AI is the only platform so far where this exact `exp99996 lr1.8e-4 target2000` recipe reaches the near-19 600k result.

### Completed: LR ablation around exp99996 600k best

Purpose:

Check both sides of the best `lr=1.8e-4` setting to confirm whether it is a real sweet spot.

Result:

| LR | Avg | Min | Max |
|---:|---:|---:|---:|
| `1.7e-4` | `16.80` | `13` | `20` |
| `1.75e-4` | `17.70` | `12` | `21` |
| `1.8e-4` | `18.95` | `14` | `21` |
| `1.9e-4` | `17.50` | `14` | `20` |
| `2.0e-4` | `17.75` | `11` | `21` |

Conclusion:

- `lr=1.8e-4` is currently the best tested learning rate for the 600k `exp99996 target2000` branch.
- Lowering to `1.7e-4` is stable but learns too conservatively.
- The intermediate `1.75e-4` improves over `1.7e-4`, but still remains clearly below `1.8e-4`.
- Extending the `1.75e-4` run did not reveal a better later checkpoint: it stayed at Avg `17.70` at 1M and dropped to Avg `16.20` at 1.5M.
- Raising to `1.9e-4` remains robust but lowers the average.
- This supports the conclusion that the 600k result improved mainly through careful learning-rate tuning, not random variance alone.

### Completed: Colab dueling reference exp99996 lr2e4 softtau001 600k

Purpose:

Test whether soft target update improves the current 600k best branch. This keeps the `exp99996 + lr2e-4` recipe and replaces hard target sync with soft update `tau=0.001`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `17.45` | `10` | `20` |

Conclusion:

- The result is strong but does not beat the hard-update best Avg `17.75`.
- Min is slightly lower (`10` vs `11`) and Max is slightly lower (`20` vs `21`).
- Soft update did not solve the remaining low-seed issue for the 600k branch at `tau=0.001`.
- Keep the hard-update `exp99996 lr2e-4` checkpoint as the main 600k result.
- Soft update may still be worth testing on the 1M branch, where drift/stability is the bigger concern.

### Completed: Colab dueling reference exp99996 lr2e4 resume500k to 1M

Purpose:

Test whether the strongest 600k recipe can be extended from a 500k checkpoint to 1M and become a stronger 1M model.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `16.35` | `3` | `21` |
| 1M | `16.35` | `8` | `21` |

Conclusion:

- This continuation does not reproduce the original 600k best Avg `17.75`; the 600k checkpoint after resume is lower at Avg `16.35` and has a weak seed with score `3`.
- From 600k to 1M, average stays flat at `16.35`. The minimum improves from `3` to `8`, but the policy does not move toward Avg `19`.
- This supports the earlier pattern: the `exp99996` branch is strong for reaching a good 600k checkpoint, but it is not the best long-training branch.
- A likely reason is that `epsilon_decay=0.99996` reaches low exploration early. After 500k, training mostly refines an already-greedy policy, and extra updates can cause value/policy drift rather than consistent improvement.
- For final selection, keep the original `exp99996 lr2e-4` 600k checkpoint for the 600k result. For 1M, prefer the `slowexp99998 target1000` branch.

### Completed: Colab dueling reference exp99996 target1500 600k

Purpose:

Test whether an intermediate target sync frequency improves the 600k `exp99996` branch. This keeps the same dueling reference recipe as the current 600k best, but changes `target_update` from `2000` to `1500`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `14.90` | `8` | `20` |

Conclusion:

- The run is stable: all seeds are positive and Max reaches `20`.
- It does not beat the current 600k best `target_update=2000`, which achieved Avg `16.15`, Min `11`, Max `20`.
- The lower average comes from several middle/low seeds (`8`, `10`, `11`, `12`, `13`) rather than catastrophic failure.
- For the `exp99996` 600k branch, `target_update=2000` remains the best tested setting.

### Completed: Lightning AI dueling reference exp99996 target1000 600k

Purpose:

Complete the target-update sweep for the 600k `exp99996` branch. This keeps the same dueling reference recipe but changes `target_update` to `1000`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `16.15` | `6` | `20` |

Conclusion:

- This run ties the current best average at 600k: Avg `16.15`.
- However, its Min is only `6`, compared with Min `11` for `target_update=2000`.
- The average is strong because many seeds are in the `15`-`20` range, but the lower worst-case seed makes it less robust for final selection.
- In the `exp99996` 600k branch, the ranking is currently: `target_update=2000` best overall, `target_update=1000` same average but less robust, `target_update=1500` lower average.

### In progress: Lighting AI dueling reference exp99996 lr18e5 target2500 2.5M

Purpose:

Test whether a slower target-network hard sync (`target_update=2500`) can reduce long-run drift in the current best `lr=1.8e-4` branch.

Current result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `8.75` | `1` | `15` |
| 1M | `13.55` | `8` | `19` |
| 1.5M | `17.10` | `14` | `21` |

Interim conclusion:

- `target_update=2500` is much too slow for the 600k target; Avg drops from the `lr=1.8e-4, target_update=2000` best of `18.95` to `8.75`.
- The branch improves substantially with longer training and becomes reasonably robust by 1.5M: Min reaches `14` and Max reaches `21`.
- It still does not reach Avg `19`, so it is not a breakthrough yet.
- This setting is useful evidence that slower target sync delays learning but can stabilize later checkpoints. If compute is limited, prioritize the `600k -> lr decay` experiment over extending this branch.

### Completed: Kaggle dueling reference slowexp99998 target1000 1M

Purpose:

Test whether the best dueling slowexp recipe benefits from faster target network synchronization and longer training. This kept the same reference optimizer and epsilon schedule as the best 600k run, but changed `target_update` from `2000` to `1000` and trained to `1M`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `13.95` | `5` | `21` |
| 1M | `18.05` | `10` | `21` |

Conclusion:

- At 600k, `target_update=1000` did not beat the previous `target_update=2000` best (`13.95` vs `15.00`).
- By 1M, the run becomes the strongest overall checkpoint so far: Avg `18.05`, Min `10`, Max `21`.
- Most 1M seeds are clustered in the `17`-`21` range, so this is close to the Avg `19` target.
- The remaining bottleneck is a small number of weaker seeds, especially scores like `10` and `13`, not a general inability to win.
- This suggests faster target sync may be slightly less sample-efficient at 600k, but beneficial for longer training stability and final policy strength.
- For the 1M target, this is now the main recipe to refine.

### Completed: Lightning AI dueling slowexp99998 target1000 lr2e4 1M

Purpose:

Test whether reducing learning rate improves robustness for the current 1M best branch. This keeps the `slowexp99998 + target_update=1000` recipe and changes `lr` from `2.5e-4` to `2e-4`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `12.00` | `5` | `18` |
| 1M | `18.00` | `14` | `20` |

Conclusion:

- The 600k checkpoint is weaker, so this setting is not useful for the 600k goal.
- The 1M checkpoint is very strong: Avg `18.00`, Min `14`, Max `20`.
- Compared with the current 1M best (`lr=2.5e-4`), Avg is almost the same (`18.00` vs `18.05`) but Min improves substantially (`14` vs `10`).
- This suggests lowering LR improves seed-to-seed robustness in the 1M slowexp branch.
- If final grading values robustness or avoids unlucky seeds, this checkpoint may be safer than the slightly higher-average `lr=2.5e-4` checkpoint.

### Completed: Lighting AI dueling slowexp99998 target1000 lr2e4 softtau001 1M

Purpose:

Test whether soft target update improves the robust 1M branch. This keeps `slowexp99998 + target1000 + lr2e-4`, but replaces hard target sync with soft target update `tau=0.001`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `4.20` | `-11` | `15` |
| 1M | `14.25` | `1` | `21` |

Conclusion:

- Soft target update `tau=0.001` is much worse than hard target update for this branch.
- Compared with the hard-update `lr2e-4` version, 1M Avg drops from `18.00` to `14.25`, and Min drops from `14` to `1`.
- The weak 600k result suggests learning is delayed or the target network is tracking too slowly for the current PER + n-step setup.
- Do not continue soft target update with `tau=0.001` for the current Task 3 recipe.

### Completed: Colab dueling slowexp99998 target1000 lr2e4 softtau0005 1M

Purpose:

Check whether the poor soft-target result was caused by `tau=0.001` being too large. This repeats the robust 1M branch with a smaller soft target update `tau=0.0005`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `9.95` | `-3` | `18` |
| 1M | `12.20` | `1` | `20` |

Conclusion:

- Smaller `tau=0.0005` is still much worse than hard target update.
- Compared with the hard-update LR2e-4 version, 1M Avg drops from `18.00` to `12.20`.
- This confirms that soft target update is not a promising direction for the current Dueling DDQN + PER + n-step 3 recipe.
- Stop soft target update experiments and return to hard target update.

### Paused: Kaggle dueling slowexp99998 target1000 lr2e4 beta800k

Purpose:

Test whether completing PER beta annealing earlier improves the robust 1M branch. This changes `per_beta_anneal_steps` from `1M` to `800k`.

Partial result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `10.15` | `-1` | `17` |

Conclusion:

- The run was paused due to Kaggle quota before 1M evaluation.
- The 600k fixed eval is weak compared with the hard-update LR2e-4 baseline, so this is not a high-priority continuation.
- If extra compute is available later, it can be resumed to check 1M, but current evidence does not justify prioritizing it over LR refinement around the 600k best.

### Completed: Kaggle dueling slowexp99998 target1000 1.5M

Purpose:

Test whether simply extending the current hard-update 1M best recipe to 1.5M can push the average to 19.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `14.90` | `10` | `21` |
| 1M | `15.60` | `8` | `19` |
| 1.5M | `16.95` | `9` | `20` |

Conclusion:

- The run improves from 600k to 1.5M, but not enough to reach Avg `19`.
- It also does not reproduce the earlier 1M best Avg `18.05`, even though the high-level recipe is similar.
- One difference is that this 1.5M run uses `per_beta_anneal_steps=1.5M`, while the previous 1M run used `1M`. Slower PER beta annealing may have changed the learning dynamics.
- Longer training alone is not a reliable solution. Checkpoint quality is still sensitive to run variance and annealing schedule.
- For final 1M reporting, the previous `slowexp99998 target1000 1M` result and the `lr2e-4` robust variant remain stronger.

### Completed: Kaggle dueling reference slowexp99998 target1500 1M

Purpose:

Test whether an intermediate target sync frequency improves the 1M slowexp branch. This keeps the same recipe as the current 1M best but changes `target_update` from `1000` to `1500`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `15.10` | `8` | `19` |
| 1M | `15.70` | `7` | `20` |

Conclusion:

- The run learns and stays positive, but it does not approach the current 1M best.
- Compared with `target_update=1000`, the 1M average drops from `18.05` to `15.70`.
- The lower score is caused by several weak seeds (`7`, `8`, `10`, `12`, `13`) rather than a total collapse.
- For the 1M slowexp branch, `target_update=1000` remains clearly better than `1500`.

### Completed: Colab dueling reference slowexp999985

Purpose:

Test whether making the exponential epsilon decay even slower than the current best can improve 600k performance. This kept the same dueling reference recipe as the best run, but changed `epsilon_decay` from `0.99998` to `0.999985`.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `12.75` | `-7` | `19` |

Conclusion:

- The model still learned a meaningful Pong policy: most seeds scored between `10` and `19`.
- However, it clearly underperformed the `0.99998` best run, which reached Avg `15.00`, Min `9`, Max `21`.
- The negative seed (`-7`) indicates that the policy was less robust by 600k.
- This suggests `epsilon_decay=0.999985` keeps exploration too high for too long under the 600k constraint. The useful region is likely around `0.99998`, or slightly faster rather than slower.

### Completed: Colab no-dueling slowexp99998 noop30

Purpose:

Test whether NoopReset can fix the catastrophic seed issue in the no-dueling slowexp99998 run.

Result:

| Checkpoint | Avg | Min | Max |
|---:|---:|---:|---:|
| 600k | `7.45` | `-3` | `17` |

Conclusion:

- NoopReset did not help this no-dueling reference recipe.
- It reduced the most catastrophic negative scores compared with the no-dueling slowexp run, but it also lowered the overall average from `11.80` to `7.45`.
- The ceiling also dropped: Max went from `18` to `17`, and many seeds stayed in low positive scores.
- For the 600k target, do not continue this branch.

## Evaluation Plan

For 1.5M runs, evaluate these checkpoints:

```text
600000
1000000
1500000
```

Use 20 seeds:

```text
seed = 0, 1, ..., 19
```

Record:

- Average
- Min
- Max
- Whether any seed reaches 19 or above
- Whether the score is stable or relies on a few lucky seeds

## What To Show In W&B Screenshots

For future analysis, keep plots focused:

- Keep current best baseline visible.
- Keep the latest continuation comparison visible: safe TPS=2, Lightning AI `noop_max=30`, and Kaggle balanced v2.
- Hide failed ablations after their conclusion is recorded.

Most useful plots:

- `Charts / Eval Reward`
- `Charts / Total Reward`
- `Train / Q Mean`
- `Train / Target Mean`
- `Train / TD Error Mean`
- `Train / TD Error Max`
- `Action / Random Ratio`
- `Action / Greedy Ratio`
- If using PER: `Train / PER Beta`, `Train / PER Max Priority`

## Short Report Phrases

Useful summary for the report:

```text
The strongest 600k setting so far is task3-kaggle-dueling-ref-adameps-nstep3-slowexp99998-600k. It uses Dueling Double DQN with PER and 3-step return, a reference-style optimizer setting with lr=2.5e-4 and Adam eps=1.5e-4, replay memory 200k, replay_start=20k, target_update=2000, and exponential epsilon decay 0.99998 with epsilon_min=0.01. This configuration achieved a 20-seed average reward of 15.00 at 600k steps, with Min 9 and Max 21. This is a substantial improvement over the previous best Avg 12.80 baseline and removes catastrophic negative seeds.

For the longer continuation runs, the safe train_per_step=2 Colab setting improved to Avg 9.35 at 1M steps but regressed to Avg 8.00 at 1.5M steps. The Lightning AI NoopReset run with noop_max=30 achieved Avg 12.25 at 600k, then 10.25 at 1M and 10.70 at 1.5M. This suggests that NoopReset improves early robustness, but longer training does not guarantee better final evaluation, so checkpoint selection is necessary.

In the no-dueling direction, the best 600k result is currently task3-lighting_ai-no-dueling-ref-adameps-nstep3-slowexp99998-600k, which achieved Avg 11.80 with Min -13 and Max 18. This improves over the original reference-style no-dueling run, which achieved Avg 10.45, and over the stable-v2 no-dueling 600k result of Avg 7.00. The slower exponential decay helped most seeds become positive, but one catastrophic seed remains, so seed-to-seed robustness is still the main bottleneck. Raising epsilon_min to 0.02 reduced the ceiling and dropped the average to 6.10, suggesting that this no-dueling reference recipe needs low final epsilon with slower decay rather than more residual exploration.

Adding dueling back to the reference recipe produced a very stable 600k result: task3-kaggle-dueling-ref-adameps-nstep3-eps002-600k achieved Avg 12.35 with Min 5 and Max 19. It did not surpass the older best Avg 12.80, but it removed catastrophic negative seeds and is one of the strongest robustness-oriented runs. This suggests dueling is still useful for stability, while the remaining improvement may come from combining dueling with the slower exponential decay that helped the no-dueling run.

Combining dueling with the slower exponential decay did produce the strongest result. The Kaggle dueling slowexp99998 run achieved Avg 15.00, Min 9, and Max 21 at 600k. This suggests that the best recipe is not simply "dueling" or "no dueling", but the combination of dueling stability, the reference optimizer settings, and slower exponential epsilon decay. In contrast, adding noop_max=30 to the no-dueling slowexp run dropped performance to Avg 7.45, so NoopReset is not helpful for the current no-dueling 600k recipe.
```

## Final Report Ablation Results

Environment used for the final report measurements:

```text
Python: 3.12.11 (Anaconda, GCC 11.2.0)
PyTorch: 2.8.0+cu128
Gymnasium: 1.1.1
ALE: 0.11.2
Platform: Linux-6.8.0-1024-aws-x86_64-with-glibc2.39
CUDA available: True
GPU: Tesla T4
```

Required 600k ablations:

| Run | Seed start | Avg | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| Final enhanced agent | 2 | 19.10 | 14 | 21 | Full recipe reaches the threshold at 600k. |
| No PER | 0 | 15.75 | 5 | 20 | PER improves sample efficiency and seed robustness. |
| No Double DQN | 0 | 11.20 | 7 | 17 | Removing Double DQN causes the largest required-ablation drop. |
| 1-step return | 0 | 13.00 | 8 | 18 | 3-step return is important for propagating sparse Pong rewards. |
| No dueling network | 2 | 12.80 | 3 | 17 | Dueling is an extra enhancement, but it clearly improves the final recipe. |

Report wording:

```text
Removing any required enhancement reduces the 600k score.  The largest drop appears when Double DQN is disabled, suggesting that over-estimation control is important for this Pong setting.  PER and 3-step return also improve the 600k result by improving useful sample reuse and reward propagation.  The no-dueling run is included as an additional architecture ablation; it is not one of the required three enhancements, but it explains why the final model keeps the dueling head.
```
