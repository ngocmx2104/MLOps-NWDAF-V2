# Phase 8a — Measurement Foundation + ML Test Score — DESIGN SPEC

> **Trạng thái:** Đã brainstorm + chốt: decompose P8→(P8a foundation, P8b experiments); harness **subprocess + instrument ngoài** (Q2); ML Test Score **manifest evidence + auto-verify** (Q3). Chờ HV review trước `writing-plans`.
> **Ngày:** 2026-06-09. **Cơ sở:** `docs/THESIS_SPEC.md` §4 (experiments + protocol §4.4), `docs/research/METHODOLOGY_GROUNDING.md` §B (đối chứng) + §F (6 metrics), `docs/IMPLEMENTATION_PLAN.md` P8, `docs/HANDOFF.md` §9 (nợ 6).
> **Vị trí:** P8a = **hạ tầng đo lường tái dùng, unit-testable**. P8b (sau) = chạy Exp-1..6 + raw_data + result tables. Tiền đề: P0–P7 merged + pushed, CI/CD xanh.

---

## 1. Mục tiêu & ranh giới

**Mục tiêu:** Xây hạ tầng để **đo có đối chứng** giá trị MLOps (RQ3) một cách **tự động + tái lập + không chấm tay** — sẵn sàng cho P8b nạp experiment thật.

**Trong P8a:** harness chạy N-run (subprocess), 6 collector metric (parameterized), thống kê (Wilcoxon + bootstrap CI), ML Test Score assessor (manifest + auto-verify), provenance logging (§9.6). Mọi đơn vị **unit-test trên input synthetic/fixture** — KHÔNG cần experiment thật.

**Ngoài P8a (→ P8b):** Exp-1..6 runners · `raw_data` full run (~60min) · giá trị C(FP)/C(FN) + biện minh · sinh DORA events thật · nợ §9.7/§9.8/§9.14/§9.9 · result tables luận văn.

**Anti-pattern guard (CLAUDE.md §3):** P8a **không sinh số liệu kết quả** (chỉ hạ tầng) → không có gì để bịa. Hai vùng "chấm tay" (Maturity + Business Impact) được khử chủ quan ngay ở thiết kế collector (ML Test Score auto-verify + expected-cost đếm được).

---

## 2. Quyết định đã chốt

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| **Q1** | Decompose | **P8a foundation → P8b experiments** | P8 quá lớn cho 1 spec/plan; foundation unit-testable, P8b phụ thuộc |
| **Q2** | Harness arch | **Subprocess + instrument ngoài + đọc artifact** | Resource/overhead đo trung thực, core sạch (backend-agnostic), cô lập per-run |
| **Q3** | ML Test Score | **Manifest evidence + assessor auto-verify** | Mỗi điểm truy về artifact machine-checkable; tái lập; chống "chấm tay" (thầy Trình) |
| **Stats** | Significance + CI | **Wilcoxon signed-rank (α=0.05) + bootstrap CI95 + mean±std** | Paired C0-vs-C1, không giả định chuẩn (METHODOLOGY §B.3) |
| **§9.6** | Provenance | **Fold vào P8a** | Reproducibility metric + delta traceability C1-vs-C0 cần lineage |

---

## 3. Cấu trúc package `src/experiments/`

Theo convention repo (mỗi subpackage có `schema.py` + `cli.py` + `__main__.py`):

```
src/experiments/
  __init__.py
  schema.py        ExperimentConfig, RunConfig (seeds, n_runs, warmup_drop, backend, workload cmd)
  runner.py        run_n(...) — N-run subprocess loop + seed mgmt + drop warmup
  records.py       raw JSONL + result-summary + evidence-pack (port từ harness cũ, reframe C0/C1)
  stats.py         wilcoxon_compare(), bootstrap_ci(), summarize()
  maturity.py      MLTestScore assessor (manifest load + auto-verify + MIN + Google/Azure derive)
  metrics/
    __init__.py
    model_perf.py      P/R/F1/ROC-AUC/PR-AUC/MCC từ (y_true, y_score)
    operational.py     timing, latency p50/p95/p99, DORA (từ event-log)
    business.py        expected_cost = FP·C(FP)+FN·C(FN), detections/hr, early-detection lead
    drift_quality.py   wrap src/monitoring PSI/KS + null%/dup/schema
    resource.py        psutil RSS/CPU + tracemalloc + storage size + tracking-overhead Δ
  cli.py · __main__.py
  mltest_manifest.yaml   28-test evidence manifest (trong src/experiments/)
tests/experiments/...  unit tests mỗi collector + runner + stats + maturity (input synthetic)
```

**Nguyên tắc isolation:** mỗi file 1 trách nhiệm; collector là hàm thuần `(inputs) -> dict[metric]`, không side-effect ngoài đọc artifact/đo process → test độc lập dễ.

---

## 4. Harness (`runner.py`)

- **Một run = một subprocess** (workload cmd từ `RunConfig`, vd `python -m src.training train ...`). Harness:
  - đo `time.perf_counter()` (wall-time),
  - sample `psutil` RSS/CPU peak trên child process,
  - đọc artifact pipeline emit (`training_result.json`, `predictions.jsonl`, `drift_history.jsonl`) để lấy ML/drift metric.
- **Protocol (METHODOLOGY §B.3 / THESIS_SPEC §4.4):** `n_runs ≥ 10`, **fixed seeds** (list cố định trong config), **drop warmup** (loại run đầu khỏi thống kê), ghi **raw JSONL mỗi run** → `artifacts/experiments/<exp>/runs.jsonl`.
- **Tracking-overhead Δ:** helper chạy cùng workload với `MLOPS_BACKEND=noop` vs `mlflow`, trả Δ(timing) + Δ(RSS) — lõi cho Exp-2/3 ở P8b.
- **Determinism:** seeds truyền vào workload qua arg/env; harness KHÔNG dùng `Date.now`/random (tái lập).

---

## 5. 6 collector metric (parameterized, unit-test trên synthetic)

| Module | Đo gì / từ đâu | Unit-test |
|---|---|---|
| `model_perf.py` | Precision/Recall/F1, **ROC-AUC**, **PR-AUC**, MCC (sklearn) từ `(y_true, y_score)` | confusion-matrix/score biết trước → giá trị kỳ vọng |
| `operational.py` | timing (perf_counter), latency p50/p95/p99 (từ `predictions.jsonl`), **DORA** (lead time/deploy freq/change-fail/MTTR) compute từ **event-log** (`deployment_history.jsonl`+`retrain_history.jsonl`) | event-log fixture → DORA biết trước |
| `business.py` | **expected_cost = FP·C(FP)+FN·C(FN)**, detections/giờ, early-detection lead time. **C(FP)/C(FN) tham số hóa** (giá trị + biện minh Elkan 2001 đặt ở P8b/thesis) | confusion + cost biết trước |
| `drift_quality.py` | PSI 3 tầng + KS (wrap `src/monitoring/psi`,`detector`) + null%/dup/schema rate | reuse test monitoring + fixture |
| `resource.py` | `psutil` RSS/CPU, `tracemalloc` peak, storage overhead (bytes), **tracking-overhead Δ** | subprocess giả (vd `python -c sleep/alloc`) |
| `maturity.py` | ML Test Score (xem §6) | manifest tí hon (pointer thật + giả) |

> Tất cả collector **build ở P8a** dưới dạng hàm tái dùng + test; **P8b** chỉ nạp dữ liệu experiment thật vào.

---

## 6. ML Test Score assessor (`maturity.py`) — manifest + auto-verify

**Manifest** `src/experiments/mltest_manifest.yaml`: 28 test (Breck 2017) × 4 mục {Data, Model, Infrastructure, Monitoring}, mỗi test:
```yaml
- id: data_1_feature_expectations
  section: data
  score: 1            # 0 / 0.5 (manual) / 1 (automated)
  evidence:
    - kind: pytest    # pytest | path | workflow | symbol
      ref: tests/features/test_quality.py::test_d2_checks
```

**`assess(manifest, repo_root)`:**
1. Với mỗi test, **verify từng evidence pointer**: `path` → file tồn tại; `pytest` → test node tồn tại (collect) + (tùy chọn) pass; `workflow` → job có trong `.github/workflows`; `symbol` → codegraph/grep symbol tồn tại.
2. **Chỉ tính điểm khi evidence verify được** (score 1 đòi pointer `pytest`/`workflow` tự động; 0.5 đòi `path`/thủ tục; 0 = không/đỏ).
3. Tổng từng mục → **điểm = MIN(4 mục)** (Breck).
4. Emit **report JSON**: điểm mỗi test + evidence đã verify + tổng/MIN + **Google L0–2 & Azure L0–4 suy ra** (present/absent từ cùng evidence).

**Unit-test:** manifest giả với pointer tồn tại (→ tính điểm) và pointer giả (→ không tính) → khẳng định MIN + report đúng. **Chống chấm tay:** điểm không verify được = không được tính.

---

## 7. Provenance (§9.6) — fold P8a

`run_training` (iforest path) **đã tính** `TrainingContext` (`ctx`: `dataset_id`/`feature_version`/`source_snapshot_id`) nhưng CHƯA log (§9.6) → **log `ctx` dạng params/tags + `dvc_rev` pointer**; **KHÔNG** log full parquet mỗi run (N≥10 → tốn storage). Test: tracker noop nhận đủ tag; mlflow-sqlite lưu được. Phục vụ reproducibility metric + delta traceability C1-vs-C0 (RQ3). (LSTM path tách rời §9.7 → provenance LSTM để P8b cùng §9.7.)

---

## 8. Thống kê (`stats.py`)

- `wilcoxon_compare(c0_samples, c1_samples)` → `scipy.stats.wilcoxon` (paired, α=0.05) trả statistic + p-value.
- `bootstrap_ci(samples, ci=0.95, n_boot=10000, seed=...)` → CI không giả định chuẩn (seed cố định để tái lập).
- `summarize(samples)` → mean±std + n + CI.
- Unit-test trên mẫu biết trước (vd 2 phân phối tách rõ → p<0.05).

---

## 9. Truy vết RQ → component → metric (chống "đồ án ráp công cụ")

| Component P8a | RQ | Vai trò |
|---|---|---|
| runner + 6 collectors | RQ3 | hạ tầng đo 6 nhóm metrics khách quan |
| stats (Wilcoxon/CI) | RQ3 | kết luận có ý nghĩa thống kê (N≥10) |
| ML Test Score assessor | RQ3 | maturity khách quan, evidence-based |
| provenance §9.6 | RQ3 | reproducibility + delta traceability C1-vs-C0 |
| tracking-overhead Δ helper | RQ3 | lõi đo Exp-2/3 (P8b) |

---

## 10. Chiến lược test (TDD, unit-only ở P8a)

Mọi đơn vị có unit-test trên **input synthetic/fixture** (P8a KHÔNG chạy experiment thật):
- collectors: input biết trước → output kỳ vọng.
- runner: workload giả (vd `python -c "..."`) → đo được timing/RSS + ghi JSONL đúng schema; n_runs/warmup-drop đúng.
- stats: mẫu biết trước → p-value/CI đúng.
- maturity: manifest giả (pointer thật+giả) → điểm/MIN/report đúng.
- provenance: tracker noop/mlflow-sqlite nhận tag đúng.
Full `pytest` + `ruff` xanh; không phá 133 test cũ.

---

## 11. Definition of Done (kiểm chứng được)

1. `src/experiments/` package đầy đủ (schema/runner/records/stats/maturity/metrics/cli) + `python -m src.experiments` chạy.
2. 6 collector có unit-test xanh trên synthetic; output schema ổn định.
3. `runner.run_n` chạy workload giả qua subprocess: đo timing+RSS, ghi raw JSONL, drop warmup, n_runs đúng. ✔ test.
4. `stats` Wilcoxon + bootstrap CI + summarize ✔ test trên mẫu biết trước.
5. ML Test Score assessor: manifest + auto-verify + MIN + Google/Azure derive ✔ test (pointer thật/giả). `src/experiments/mltest_manifest.yaml` khởi tạo (điền thật ở P8b khi có đủ artifact).
6. Provenance §9.6: lineage tags logged ✔ test.
7. Toàn bộ test cũ (133) + mới xanh; `ruff` sạch.
8. Truy vết RQ/metric (§9) + ranh giới P8a/P8b (§1) rõ; KHÔNG sinh số liệu kết quả ở P8a.

> **Quy tắc vàng:** P8a là hạ tầng đo — KHÔNG bịa số liệu; mọi collector gắn 1 nhóm metric khách quan; ML Test Score chỉ tính điểm có evidence verify được. Số liệu thật sinh ở P8b.
