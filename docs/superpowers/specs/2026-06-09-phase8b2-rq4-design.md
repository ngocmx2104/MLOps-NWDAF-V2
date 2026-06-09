# Phase 8b-2 — RQ4: Linh hoạt (Exp-4 drift+retrain, Exp-5 model-swap, Exp-6 modifiability) — DESIGN SPEC

> **Trạng thái:** Đã brainstorm + HV duyệt (06/2026). Quyết định HV: (1) **coi synthetic drift là data thật** (skip raw_data); (2) **Exp-6 = apply mod thật + git diff + regression**; (3) Exp-6 chi tiết + (4) tiêu chí RQ4 = quyền quyết của agent. Chờ HV review spec → `writing-plans`.
> **Ngày:** 2026-06-09. **Cơ sở:** `docs/THESIS_SPEC.md` §4, `docs/research/METHODOLOGY_GROUNDING.md` §B/§F, hạ tầng **P8a** (`src/experiments/`), **P6 monitoring** (`DriftDetector`, `AutoRetrainTrigger`, `run_retrain_cycle`), **P4 training** (`run_training`, IForest + LSTM-AE), **P8b-1** (`exp_common`, `tables`, `cli`).
> **Tiền đề:** P0–P8b-1 merged trên `main`. Có sẵn: synthetic drift features `artifacts/synthetic/features/features_drift_{sudden,gradual,recurring}.parquet` + baseline seeds; real D5.

---

## 1. Mục tiêu & ranh giới

**Mục tiêu:** **Sinh số liệu thật** trả lời **RQ4** — *"Pipeline có đủ linh hoạt khi (a) dữ liệu trôi dạt → tự retrain, (b) thay model, (c) đổi feature/giám sát — mà không phá quy trình?"* — đo tự động, tái lập, **có tiêu chí pass/fail khách quan** (chống "đủ linh hoạt" mơ hồ).

**Trong P8b-2:**
- **Exp-4** — Drift detection (PSI 3 tầng + KS/Evidently) + **closed-loop auto-retrain** trên 3 kịch bản sudden/gradual/recurring; đối chứng **loop ON vs OFF**.
- **Exp-5** — Model-swap **IForest ↔ LSTM-AE** qua cùng `run_training`; **so sánh định lượng 2 model** (headline) + tính linh hoạt (swap không chạm core).
- **Exp-6** — Modifiability: 3 mod đặc thù handover, đo **git diff footprint + regression** (khách quan, tái lập).
- **Result tables** — mở rộng `tables.py` cho Exp-4/5/6.

**Ngoài P8b-2 (→ P9 luận văn):** viết chương kết quả; bảng LaTeX hoàn chỉnh; `raw_data` full (không cần cho RQ4 — đã quyết skip).

**Anti-pattern guard (CLAUDE.md §3) — các điểm phản biện đã lường:**
- **Exp-4:** coi synthetic là data thật (HV quyết); recovery ON/OFF có nhãn → loop có *giá trị đo được*, không chỉ "trigger fired".
- **Exp-5:** headline = **so sánh định lượng** (không phải "0 dòng code" — tránh "kết luận hiển nhiên" #2). LSTM-AE biện minh = chuỗi handover/IMSI; nếu không thuyết phục → khai báo là *phép thử modifiability*.
- **Exp-6:** **regression = metric chính** (file/dòng là proxy phụ); mod đặc thù NWDAF (giữ đúng đề tài — lỗi Cô Thủy); footprint diễn giải trung thực.
- **Mọi Exp:** tiêu chí pass/fail định trước (§7) → KHÔNG kết luận cảm tính.

---

## 2. Quyết định đã chốt

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| **Q1** | Dữ liệu drift | **synthetic đã có** (skip raw_data) | HV quyết coi synthetic là thật; đủ cho RQ4, không tốn 60min |
| **Q2** | Exp-6 đo | **apply mod thật + git diff + regression** | khách quan, tái lập, chống "chấm tay" (thầy Trình) |
| **Q3** | Exp-4 window | **harness tự quản scenario clock, file prediction/step** | fix §9.14 (DriftDetector đọc tích lũy → pha loãng PSI) |
| **Q4** | Exp-5 split | **fix §9.7** — LSTM dùng `load_training_dataset`+`cfg.test_size`+`FEATURE_COLUMNS` | so sánh 2 model **cùng split** (công bằng) |
| **Q5** | Exp-6 an toàn | **git worktree tạm** | apply→đo→bỏ, KHÔNG bẩn repo chính |
| **Q6** | Exp-5 headline | **so sánh định lượng 2 model** | tránh anti-pattern #2 "hiển nhiên" |

---

## 3. Cấu trúc package (mở rộng `src/experiments/`)

```
src/experiments/
  exp4_drift.py        🆕 scenario-clock harness: step qua baseline→drift, file pred/step, detect+retrain, ON/OFF
  exp5_model_swap.py   🆕 IForest vs LSTM-AE qua run_training (cùng dataset/split), so sánh + swap-footprint
  exp6_modifiability.py🆕 apply 3 mod handover trong worktree → git diff --stat + pytest regression → revert
  modifications/       🆕 3 mod đặc thù NWDAF (patch/scripted edit + metadata: section, kỳ vọng)
  tables.py            ✏️ thêm exp4_table / exp5_table / exp6_table
  cli.py               ✏️ thêm run-exp4 / run-exp5 / run-exp6
src/training/          ✏️ §9.7 fix: lstm_detector + pipeline thread cfg.test_size, FEATURE_COLUMNS chung
tests/experiments/...  🆕 unit test tiny-N/fixture mỗi harness
artifacts/experiments/<exp>/  output chạy thật (gitignored)
```

---

## 4. Exp-4 — Drift detection + auto-retrain (RQ4a)

**Scenario clock (fix §9.14):** harness điều khiển đồng hồ kịch bản, **mỗi step ghi file prediction riêng** (KHÔNG tích lũy → PSI không bị pha loãng):
1. **reference** = baseline window `feature_values` (drift là **covariate** trên feature → reference = phân phối feature baseline, KHÔNG cần score bằng model).
2. **Stepping:** nạp scenario `features_drift_{sudden,gradual,recurring}` theo **window cố định** từng bước (baseline steps → drift onset → post-drift). Mỗi step:
   - score window bằng model hiện tại → `step_<k>_predictions.jsonl` (có `feature_values` cho drift, `anomaly_score`).
   - `DriftDetector.detect(reference, step_<k>_predictions)` → PSI 3 tầng + Evidently.
   - `AutoRetrainTrigger.should_retrain(drift)` → nếu **loop ON** & drift: `run_retrain_cycle(...)` trên window gần nhất → reload model → tiếp.
3. **Persistent `AutoRetrainTrigger`** xuyên step → `retrain_count` tích lũy đúng.

**Đo:**
- **Detection latency** = số step từ drift onset tới khi flag (per kịch bản).
- **PSI/Evidently** per step (đường drift theo thời gian).
- **retrain_count** + step retrain kích hoạt.
- **ON vs OFF (giá trị closed-loop):** weak-label drift window (rule `apply_weak_labels` đã có) → đo model performance (ROC-AUC) **trước drift / sau drift không-retrain (OFF) / sau drift có-retrain (ON)**. Kỳ vọng: OFF tụt, ON hồi phục.

**Output:** `exp4_summary.json` = per-scenario {detection_latency, psi_curve, retrain_count, perf_before/after_off/after_on}.

---

## 5. Exp-5 — Model-swap IForest ↔ LSTM-AE (RQ4b)

**§9.7 fix (Q4):** sửa `train_lstm_ae` (+ pipeline) để LSTM dùng **chung** `load_training_dataset`/`prepare_training_matrices`/`split_training_data` + `cfg.test_size` + `FEATURE_COLUMNS` như IForest → **cùng train/val split** (so sánh công bằng). Có test guard chống drift FEATURE_COLUMNS (đã có).

**Biện minh LSTM-AE:** 7 feature/(IMSI,window) là vector tĩnh → LSTM cần chiều tuần tự. **Khung dùng:** chuỗi các handover-window của **cùng 1 IMSI theo thời gian** (sequence) cho LSTM-AE reconstruct; nếu dữ liệu/khung không đủ chuỗi → **khai báo trung thực** LSTM-AE là *ứng viên thay thế để KIỂM CHỨNG tính swap*, IForest là model production chính (đúng THESIS_SPEC).

**Đo (headline = so sánh):**
- **Quality:** ROC-AUC/PR-AUC/F1 mỗi model (N=10 seeds, cùng dataset/split). Wilcoxon nếu so 1 metric.
- **Operational:** train-time, resource (qua harness P8a).
- **Swap footprint (phụ):** số dòng/file phải đổi để swap = **0 ở core** (chỉ cờ `model_type`) — bằng chứng linh hoạt, KHÔNG phải headline.

**Output:** `exp5_summary.json` = {iforest: {quality, train_s}, lstm_ae: {quality, train_s}, swap_core_changes: 0}.

---

## 6. Exp-6 — Modifiability (RQ4c)

**3 mod đặc thù handover (định nghĩa trước):**
- **Mod-A (data):** thêm 1 feature handover thứ 8 (vd `max_consecutive_pingpong`) — chạm `features/schema.py` + `builder.py`.
- **Mod-B (label):** đổi rule weak-label ping-pong (vd `pingpong_count≥1` → `≥2`) — chạm `weak_labels.py`/`schema.py`.
- **Mod-C (monitoring):** đổi ngưỡng PSI drift (vd `psi_alert 0.25→0.20`) — chạm `monitoring/schema.py`.

**Đo (Q2, Q5 — khách quan, an toàn):** với mỗi mod, trong **git worktree tạm**:
1. apply mod (scripted edit định nghĩa trong `modifications/`).
2. **`git diff --stat`** → #file chạm, #dòng +/− (proxy phụ).
3. **`pytest`** → **regression count** = #test fail (metric **chính**).
4. bỏ worktree (repo chính sạch).

**Tiêu chí + diễn giải:** footprint nhỏ + 0 regression = bằng chứng kiến trúc modular (schema-driven, BaseTracker tách lớp). **Trung thực:** đây là footprint *tuyệt đối* trên kiến trúc C1; diễn giải định tính "ad-hoc script sẽ phải sửa rải rác + không có test bắt regression" (KHÔNG dựng baseline ghép-cứng riêng — ghi rõ là interpretation, không phải số đo).

**Output:** `exp6_summary.json` = per-mod {section, files_changed, lines_changed, regression_count, pass}.

---

## 7. Tiêu chí pass/fail RQ4 (định trước — chống "đủ linh hoạt" mơ hồ)

| Sub-RQ | Exp | Tiêu chí PASS (khách quan) |
|---|---|---|
| RQ4a drift | Exp-4 | drift **detect được** ở cả 3 kịch bản (PSI≥alert); auto-retrain **kích hoạt đúng** (chỉ khi drift); ON **hồi phục** perf hơn OFF |
| RQ4b swap | Exp-5 | swap **0 thay đổi core** (chỉ cờ); cả 2 model train+eval **chạy qua cùng pipeline**; ra metric so sánh được |
| RQ4c modify | Exp-6 | mỗi mod: **0 regression** (test cũ vẫn xanh sau mod); footprint **giới hạn ở lớp liên quan** (không lan ra core/training/serving không liên quan) |

> Báo cáo pass/fail per tiêu chí — KHÔNG tự cho điểm "linh hoạt".

---

## 8. Chiến lược test (TDD)

- **Harness unit-test (tiny-N/fixture):** exp4 (kịch bản giả 2-step + model giả → verify scenario-clock ghi file/step, detect, retrain count, ON/OFF nhánh); exp5 (tiny parquet, iforest+lstm qua run_training → verify 2 nhánh chạy + summary); exp6 (1 mod giả trong worktree tạm → verify git diff + regression đếm + revert sạch). **KHÔNG khẳng định số khoa học trong test.**
- **§9.7 fix:** test LSTM dùng cùng split (so sánh n_train/n_val với IForest path).
- **Chạy thật:** sinh `exp{4,5,6}_summary.json` (artifact gitignored) = số luận văn.
- Full `pytest` + `ruff` xanh; KHÔNG phá test cũ.

---

## 9. Truy vết RQ → component → metric

| Component | Sub-RQ | Vai trò |
|---|---|---|
| Exp-4 (drift+retrain ON/OFF) | RQ4a | pipeline tự phát hiện drift + retrain (giá trị closed-loop) |
| Exp-5 (IForest vs LSTM-AE) | RQ4b | linh hoạt thay model + so sánh định lượng 2 họ |
| Exp-6 (mod footprint+regression) | RQ4c | modifiability khách quan (kiến trúc modular) |
| tiêu chí pass/fail (§7) | RQ4 | kết luận có ngưỡng, không cảm tính |

---

## 10. Definition of Done (kiểm chứng được)

1. `exp4_drift` + `exp5_model_swap` + `exp6_modifiability` + mở rộng `tables`/`cli` đầy đủ; `run-exp4/5/6` chạy.
2. §9.7 fix: LSTM cùng split với IForest ✔ test.
3. Harness mỗi Exp unit-test xanh trên tiny-N/fixture (verify cơ chế, KHÔNG số khoa học).
4. **Chạy thật:** `exp{4,5,6}_summary.json` số thật — Exp-4 (3 kịch bản: latency+retrain_count+ON/OFF perf); Exp-5 (so sánh 2 model + swap core=0); Exp-6 (3 mod: footprint+regression).
5. Tiêu chí pass/fail RQ4 (§7) báo cáo per Exp.
6. tables exp4/5/6 sinh từ summaries (0 hard-code).
7. Test cũ + mới xanh; ruff sạch; worktree Exp-6 không bẩn repo.
8. Truy vết RQ4 đầy đủ; mod đặc thù handover; **0 số bịa**; ranh giới P8b-2/P9 rõ.

> **Quy tắc vàng:** Exp-5 headline = so sánh định lượng (không "0 dòng hiển nhiên"); Exp-6 regression = chính; tiêu chí RQ4 định trước; mọi số từ code chạy được + tái lập; chưa chạy → "N/A", KHÔNG bịa.
