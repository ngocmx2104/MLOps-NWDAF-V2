# Phase 8b-1 — RQ3 spine (Exp-1, Exp-2, Exp-3 + ML Test Score 28-test) — DESIGN SPEC

> **Trạng thái:** Đã brainstorm + HV duyệt (06/2026). Quyết định: decompose P8b → **P8b-1 (RQ3 spine, spec này)** + P8b-2 (RQ4: Exp-4/5/6 + raw_data full). Chờ HV review spec → `writing-plans`.
> **Ngày:** 2026-06-09. **Cơ sở:** `docs/THESIS_SPEC.md` §4 (thực nghiệm + protocol §4.4), `docs/research/METHODOLOGY_GROUNDING.md` §B (đối chứng) + §F (6 metrics), hạ tầng **P8a** (`src/experiments/`: `runner`, 6 collector, `stats`, `maturity`, `records`).
> **Tiền đề:** P0–P8a merged trên `main` (152 test xanh). Có sẵn: real D5 (8.393 UE-window, weak-label 19.49%), trained iforest, MLflow local (sqlite), ClearML offline.

---

## 1. Mục tiêu & ranh giới

**Mục tiêu:** **Sinh số liệu thật, có đối chứng** trả lời **RQ3** — *"MLOps mang lại giá trị đo được gì so với ad-hoc, và framework nào phù hợp?"* — bằng 3 thực nghiệm + maturity, mọi số đo tự động + tái lập + thống kê (Wilcoxon/CI), **KHÔNG chấm tay, KHÔNG bịa**.

**Trong P8b-1:**
- **Exp-1** — Pipeline E2E trên EBS handover thật; chất lượng phát hiện ping-pong (ROC-AUC/PR-AUC/F1@thr) + timing baseline. (RQ1, RQ2 — chứng minh pipeline GIẢI ĐƯỢC bài toán.)
- **Exp-2** — Capability ablation **C0 (không MLOps) vs C1 (full MLOps/MLflow)** trên 6 nhóm metrics; **lượng hoá đánh đổi chi phí↔lợi ích**, KHÔNG tuyên bố "thắng". (RQ2, RQ3 — xương sống.)
- **Exp-3** — So sánh framework **Noop/MLflow/ClearML** ở lát tracking-registry; overhead/resource/governance; **biện minh chọn MLflow**. (RQ3.)
- **ML Test Score** — điền manifest 5→**28 test Breck**, assess → maturity khách quan (so C0 vs C1). (RQ3.)
- **Result tables** — đọc `*_summary.json` → bảng Markdown/LaTeX (mean±std + CI95 + p-value) cho luận văn.

**Ngoài P8b-1 (→ P8b-2):** Exp-4 (drift+retrain), Exp-5 (model-swap), Exp-6 (modifiability); `raw_data` full (~60min) + lock `dvc.lock`; nợ §9.7/§9.8/§9.14.

**Anti-pattern guard (CLAUDE.md §3):**
- **#1 không chấm tay:** mọi số sinh từ collector P8a (đo `perf_counter`/`psutil`/sklearn) + ML Test Score auto-verify. Maturity dựa bằng chứng artifact, không tự gán.
- **#2 không baseline rơm / kết luận hiển nhiên:** C0 là pipeline bị *gỡ* lớp MLOps (cùng data/model/code/seed) — KHÔNG straw-man; mục tiêu = **lượng hoá đánh đổi** (overhead vs maturity/traceability), KHÔNG "C1 phát hiện tốt hơn" (model-perf là biến kiểm soát, giống hệt).
- **#5 không bịa:** test harness dùng tiny-N synthetic (không chứa số kết quả); số thật sinh từ **chạy thật N≥10**, lưu artifact, đọc vào bảng. Chưa chạy được phần nào thì ghi "chưa", không điền số.

---

## 2. Quyết định đã chốt (HV duyệt)

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| **Q1** | Decompose | **P8b-1 (RQ3) trước → P8b-2 (RQ4)** | P8b quá lớn; RQ3 là xương sống, review kết quả từng phần |
| **Q2** | Cost C(FP)/C(FN) | **Sensitivity analysis** (ratio C(FN)/C(FP) ∈ {1,2,5,10,20}, neo Elkan 2001) | KHÔNG chốt 1 số tuỳ tiện → chống "bịa số"; kết luận robust trên dải |
| **Q3** | Dữ liệu | **real D5 hiện có** (defer raw_data full → P8b-2) | Không block 60min; D5 = handover thật, đủ cho quality + ablation |
| **Q4** | Overhead basis | **Backend LOCAL** (mlflow→sqlite+local artifacts; clearml→offline), KHÔNG docker server | Tái lập cho N≥10/Wilcoxon; overhead mạng/server = note ngoài phạm vi |
| **Q5** | N runs | **N=10**/config, fixed seeds, drop 1 warmup | Protocol §4.4 (METHODOLOGY §B.3) |
| **Q6** | Harness arch | **Subprocess-orchestrated** (dựng trên `runner.run_n` P8a) | Đo resource trung thực, cô lập per-run, đúng protocol |

---

## 3. Cấu trúc package (mở rộng `src/experiments/`)

```
src/experiments/                         # đã có (P8a): schema, records, runner, stats, maturity, metrics/*, cli, __main__
  exp_common.py        🆕 ExperimentSpec (configs/N/seeds/dataset/workload) + run_config() loop (run_n → collect → summarize) + artifact reader
  exp1_e2e.py          🆕 Exp-1: quality E2E (C1) → exp1_summary.json
  exp2_ablation.py     🆕 Exp-2: C0 vs C1, 6 nhóm, Wilcoxon, cost-sensitivity → exp2_summary.json
  exp3_framework.py    🆕 Exp-3: Noop/MLflow/ClearML overhead+governance → exp3_summary.json
  tables.py            🆕 summaries → bảng Markdown + LaTeX (mean±std, CI95, p-value)
  mltest_manifest.yaml ✏️ điền 5 → 28 test Breck (mỗi test trỏ artifact thật)
  cli.py               ✏️ thêm subcommand: run-exp1 | run-exp2 | run-exp3 | tables (giữ assess)
tests/experiments/
  test_exp_common.py · test_exp1.py · test_exp2.py · test_exp3.py · test_tables.py · test_maturity_28.py   🆕 (tiny-N, fixture)
artifacts/experiments/<exp>/             # output chạy thật (gitignored): runs.jsonl + <exp>_summary.json
```

**Nguyên tắc:** Exp runner = orchestrator mỏng. Logic đo nằm ở collector P8a (đã test). Runner chỉ: dựng config → `run_n` (subprocess) → đọc artifact emit → gọi collector → `stats` → ghi summary. Mỗi runner test độc lập bằng workload giả (tiny-N).

---

## 4. Workload & artifact emission (chốt vấn đề subprocess)

- **Một run = một subprocess** (cô lập). Workload chính = `python -m src.training <args>` (đã là điểm điều khiển C0/C1/C2 qua `MLOPS_BACKEND`).
- **Vấn đề:** `run_training()` trả `dict` in-process; subprocess KHÔNG trả dict. → **Mở rộng training CLI** để ghi `training_result.json` (metrics + fit_summary + validation_summary + model_path + lineage) vào `output_dir`. Harness đọc JSON này lấy ML metrics. (Đây cũng đóng một phần nợ §9.7 "fit/validation chưa log".) Thay đổi tối thiểu, có test riêng.
- **Latency:** workload Exp-1/2 thêm bước predict (gọi `ServingRuntime` hoặc `python -m src.serving` trên N mẫu) → `predictions.jsonl` (đã có `latency_ms`) → collector `operational.latency_percentiles`.
- **Seeds:** truyền qua `--random-state {seed}` (token thay trong `runner._bind`). Determinism: cùng seed → cùng model/metrics.

---

## 5. Exp-1 — E2E + chất lượng phát hiện (RQ1, RQ2)

- **Config:** C1 (mlflow-local), N=10 seeds, dataset = real D5 (`artifacts/features_labeled/D5_*.parquet`), weak-label = cột `label`.
- **Đo:**
  - **Model-perf** (`metrics.model_perf`): ROC-AUC, PR-AUC (rank-based, không phụ thuộc ngưỡng) + F1/precision/recall/MCC **@ ngưỡng vận hành** — ngưỡng = điểm cắt từ `contamination` của IsolationForest (đại lượng model sinh ra, KHÔNG chọn tay), ghi rõ biện minh ở luận văn. Tính từ `(y_true=weak_label, y_score=anomaly_score)` trên tập validation.
  - **Operational:** pipeline wall-time (subprocess), predict latency p50/p95/p99.
- **Output:** `exp1_summary.json` = per-seed metrics + summarize (mean±std + CI95). Bảng: "chất lượng phát hiện ping-pong trên N=10 seeds".
- **Vai trò:** chứng minh **pipeline E2E giải đúng bài toán** (RQ1 đặc thù dữ liệu → RQ2 thiết kế). KHÔNG so sánh — đây là baseline năng lực.

---

## 6. Exp-2 — Capability ablation C0 vs C1 (RQ2, RQ3) — XƯƠNG SỐNG

**C0 (ad-hoc):** `MLOPS_BACKEND=noop` — pipeline chạy nhưng **gỡ** tracking/registry/lineage/monitoring (NoopTracker nuốt mọi call). **C1 (MLOps):** `MLOPS_BACKEND=mlflow` (local) — log params/metrics/artifacts, registry alias, lineage (provenance §9.6). **Cùng data/model/code/seed** → biến kiểm soát duy nhất = lớp MLOps.

**6 nhóm — phân vai rõ (chống "kết luận hiển nhiên"):**

| Nhóm | Đo | Vai trò C0-vs-C1 |
|---|---|---|
| **Model-perf** | ROC-AUC/PR-AUC/F1 | **KIỂM SOÁT** — giống hệt (cùng seed/model) → chứng minh C0 KHÔNG phải model yếu (fair comparison) |
| **Business** | expected_cost (sensitivity ratio) | **KIỂM SOÁT** — cùng confusion → cùng cost; báo cáo đường cong 1 lần (thuộc tính model) |
| **Operational** | pipeline wall-time, predict latency, **deploy-time** (load+reload model) | C1 có thể chậm hơn chút (instrument) — *chi phí* |
| **Cost/Resource** | RSS/CPU peak, **storage Δ** (mlruns/registry vs ~0) | *chi phí* MLOps — định lượng overhead |
| **Drift/Data-quality** | QC dataset (null/dup/n_rows); **capability flag** (C1 có giám sát drift / lineage check; C0 không) | *năng lực hiện diện* (present/absent, automated) |
| **Maturity** | ML Test Score (assess manifest dưới C0 vs C1) + **traceability check** (truy được prediction→model→dataset_id?) | **LỢI ÍCH** — định lượng chính (C1 ≫ C0) |

**Thống kê:** với mỗi metric đo lặp (operational/resource), **Wilcoxon signed-rank C0 vs C1** (paired theo seed, α=0.05) + mean±std + CI95.

**Kết luận hướng tới (lượng hoá đánh đổi):** *"C1 tốn thêm ΔT% thời gian / ΔS MB storage (chi phí), đổi lại maturity từ <C0_score> → <C1_score> + traceability/reproducibility mà C0 không có (lợi ích)."* — số thật điền sau khi chạy. **KHÔNG** phát biểu "C1 phát hiện tốt hơn".

**Traceability check (khách quan, tự động):** hàm kiểm tra dưới C1 có thể truy 1 prediction → run_id → model version → `dataset_id`/`source_snapshot_id` (từ lineage params §9.6) hay không; dưới C0 không có run → fail. Trả boolean + bằng chứng, KHÔNG chấm điểm tay.

---

## 7. Exp-3 — So sánh framework Noop/MLflow/ClearML (RQ3)

- **Config:** 3 backend ở **lát tracking-registry** (cùng workload training + log + register), N=10 seeds. MLflow local (sqlite+artifacts), ClearML offline, Noop = baseline 0-overhead.
- **Đo:**
  - **Overhead Δ vs Noop:** wall-time, RSS (`metrics.resource` + `tracking_overhead`), **storage** (bytes mlruns vs clearml cache).
  - **ML metrics:** trùng across backend (sanity — `metrics.model_perf` giống) → xác nhận biến kiểm soát.
  - **Governance (present/absent, automated):** registry có model versioned? alias/stage? params/metrics query được? artifact lineage? → bảng năng lực (không chấm tay).
- **Output:** `exp3_summary.json` + bảng overhead (mean±std + CI95) + ma trận governance.
- **Kết luận hướng tới:** biện minh chọn **MLflow** (vd overhead thấp hơn / governance đủ / tích hợp tốt) — số + bằng chứng thật. **KHÔNG** dựng full stack ClearML (đúng phạm vi Trục-2).
- **Gotcha (HANDOFF §4):** ClearML khởi background threads → chạy trong **subprocess** (đã là kiến trúc harness) + offline; `OutputModel.update_weights` offline lỗi → register_model bọc try/except (đã có P4).

---

## 8. ML Test Score — điền 28 test (RQ3)

- Mở rộng `mltest_manifest.yaml` từ 5 seed → **28 test Breck** (Data 7 / Model 7 / Infrastructure 7 / Monitoring 7), mỗi test {0/0.5/1} + ≥1 evidence pointer (path/workflow/pytest/symbol) **trỏ artifact THẬT** trong repo (P0–P8a đã tạo nhiều: feature QC, training pipeline, CI workflows, monitoring/drift, registry, provenance...).
- `assess` (P8a) auto-verify từng pointer → điểm = MIN 4 mục + Google/Azure derive. **Chỉ tính điểm verify được.**
- Chạy assess **2 lần**: dưới năng lực C1 (full) và C0 (giả lập gỡ infra — vd manifest C0 loại các test cần tracking/registry/CI) → Δ maturity C0 vs C1 (đầu vào Exp-2 nhóm Maturity).
- **Anti-pattern:** mỗi điểm phải trỏ artifact thật; pointer giả → 0. KHÔNG tự gán 28 điểm.

---

## 9. Result tables (`tables.py`)

- Đọc `exp{1,2,3}_summary.json` + report maturity → sinh **bảng Markdown** (cho HANDOFF/review) + **bảng LaTeX** (cho luận văn ch.4): mean±std, CI95, p-value Wilcoxon, Δ overhead, Δ maturity, governance matrix.
- KHÔNG hard-code số — đọc từ artifact. Nếu summary thiếu (chưa chạy) → in "N/A (chưa chạy)", không bịa.

---

## 10. Chiến lược test (TDD)

**Hai tầng tách bạch (then chốt chống bịa số):**
1. **Unit-test harness (tiny-N, fixture):** mỗi runner test với N=2 + workload giả nhanh (vd `python -c` ghi result JSON tí hon, hoặc real D5 cắt nhỏ) → verify *điều phối đúng* (gọi run_n, đọc artifact, gọi collector, ghi summary đúng schema, Wilcoxon chạy). **Test KHÔNG khẳng định giá trị kết quả khoa học** — chỉ khẳng định cơ chế. Xanh trong CI.
2. **Chạy thật (N=10):** sau khi harness xanh, chạy `python -m src.experiments run-exp{1,2,3}` → sinh artifact thật (`artifacts/experiments/`, gitignored). **Đây là số luận văn.** Lưu artifact + (tuỳ chọn) commit `*_summary.json` nhỏ vào repo cho reproducibility, KHÔNG commit raw runs.

Full `pytest` + `ruff` xanh; KHÔNG phá 152 test cũ.

---

## 11. Truy vết RQ → component → metric

| Component P8b-1 | RQ | Vai trò |
|---|---|---|
| Exp-1 (quality E2E) | RQ1, RQ2 | pipeline giải đúng bài toán ping-pong; baseline năng lực |
| Exp-2 (C0 vs C1, 6 nhóm + Wilcoxon) | RQ2, RQ3 | lượng hoá **đánh đổi** chi phí↔lợi ích của MLOps (xương sống) |
| Exp-3 (framework overhead+governance) | RQ3 | biện minh chọn MLflow |
| ML Test Score 28-test (C0 vs C1) | RQ3 | maturity khách quan, evidence-based |
| tables (Wilcoxon/CI/Δ) | RQ3 | kết luận có ý nghĩa thống kê cho luận văn |

---

## 12. Definition of Done (kiểm chứng được)

1. `exp_common` + `exp1_e2e` + `exp2_ablation` + `exp3_framework` + `tables` đầy đủ; CLI `run-exp1/2/3` + `tables` chạy.
2. Training CLI emit `training_result.json` (metrics/fit/validation/lineage) ✔ test.
3. Harness mỗi Exp unit-test xanh trên tiny-N/fixture (verify điều phối, KHÔNG số khoa học).
4. **Chạy thật N=10:** `exp1/2/3_summary.json` có số thật (mean±std + CI95); Exp-2 có Wilcoxon C0-vs-C1 mỗi metric đo lặp; Exp-3 có bảng overhead Δ + governance.
5. `mltest_manifest.yaml` 28 test, assess C1 & C0 → Δ maturity (mọi điểm verify được).
6. `tables.py` sinh bảng md + LaTeX từ summaries (0 hard-code số).
7. 152 test cũ + mới xanh; ruff sạch.
8. Truy vết RQ/metric đầy đủ; ranh giới P8b-1/P8b-2 rõ; **KHÔNG số bịa** (chưa chạy → "N/A").

> **Quy tắc vàng:** model-perf/business là KIỂM SOÁT (chứng minh fair, không phải "thắng"); giá trị MLOps = đánh đổi *chi phí (overhead) ↔ lợi ích (maturity/traceability/reproducibility)* định lượng được; mọi số từ code chạy được + tái lập; chưa chạy thì ghi "chưa", KHÔNG bịa.
