# IMPLEMENTATION PLAN (Master Roadmap) — MLOps NWDAF Thesis

> **Đây là roadmap tổng theo *phase*.** Mỗi phase có plan chi tiết riêng (bite-sized, TDD) trong `docs/plans/`. Cơ sở: `docs/THESIS_SPEC.md` + `docs/research/METHODOLOGY_GROUNDING.md`.
> **Chủ trương:** không bó thời gian — ưu tiên hoàn chỉnh, đúng, tái lập. Build fresh; port có chọn lọc logic đã kiểm chứng từ `MLOps_Project/src/`.

## Quy ước toàn cục (chốt cho mọi phase)

- **Python 3.11+**, quản lý deps bằng `pyproject.toml` (extras: `[lstm]` cho tensorflow, `[clearml]`, `[dev]`).
- **Repo layout:** mỗi subpackage `src/<stage>/` có `schema.py` (config dataclasses) + `cli.py` (`python -m src.<stage>`).
- **Test:** `pytest` (repo cũ KHÔNG có test — đây là nâng cấp có chủ đích, +điểm ML Test Score). Mỗi component có unit test.
- **Lint/format:** `ruff`. **Pre-commit** hook. **CI:** GitHub Actions (lint+test mọi push).
- **Tracking:** mọi logic core đi qua `BaseTracker` (`create_tracker()` đọc `MLOPS_BACKEND`), KHÔNG gọi thẳng API framework.
- **Commit:** nhỏ, thường xuyên, theo từng task (Conventional Commits).
- **Reproducibility:** fixed seeds; DVC version dữ liệu+model; MLflow log params/metrics/artifacts.

## Sơ đồ phụ thuộc & thứ tự build

```
P0 Scaffold ─► P1 Ingestion ─► P2 Synthetic ─► P3 Features+Feast ─► P4 Training+MLflow ─┬─► P5 Serving ─┐
                                                                                         └─► P6 Monitoring ┴─► P7 CI/CD ─► P8 Experiments+Maturity ─► P9 Thesis
```

## Bảng phase

| Phase | Tên | Kreuzberger | Mục tiêu (deliverable chính) | Định nghĩa Hoàn thành (DoD — kiểm chứng được) | RQ/Exp |
|---|---|---|---|---|---|
| **P0** | Scaffold & Tracking abstraction | nền tảng | Repo cài được, tooling, CI skeleton, DVC init, `BaseTracker`(Noop/MLflow/ClearML) | `pip install -e .` OK · `pytest` xanh · `ruff` sạch · CI pass · `create_tracker('noop')` test pass · `dvc status` OK | nền mọi RQ |
| **P1** | Data ingestion | C2,C3 | EBS parser + schema validation + quality + D1 snapshot; DVC stage `parse` | parse real+synthetic EBS → snapshot parquet versioned; `dvc repro parse` chạy; data-quality metrics xuất ra | RQ1 · Exp-1 |
| **P2** | Synthetic data | hỗ trợ | `real_profile` + generators + 3 kịch bản drift + datasets CLI | sinh baseline + sudden/gradual/recurring; `manifest.json` có `drift_active`; calibrate từ EBS thực | Exp-1,4 |
| **P3** | Features + Feast | C3 | 7 đặc trưng + weak labels; Feast repo (offline parquet + online); materialize | features vào Feast offline+online; **test train/serve consistency** pass; online retrieve OK | RQ2 · Exp-1 |
| **P4** | Training + MLflow | C4,C5,C6 | IForest + LSTM-AE; train pipeline log MLflow; registry alias/stage; eval ROC-AUC/PR-AUC | train→MLflow run logged→registered (alias staging→prod); C0 noop path chạy; model-swap không đổi pipeline | RQ2,3,4 · Exp-2,3,5 |
| **P5** | Serving | C7 | FastAPI `/predict` (lấy online feature từ Feast), `/health`, hot-reload, rollback; Docker | serve model registered; `/predict` trả score+latency; container chạy; rollback về version khác OK | RQ2 · Exp-1 |
| **P6** | Monitoring + auto-retrain | C8 | PSI 3 tầng + Evidently drift + system metrics (Prometheus/Grafana) + retrain trigger | phát hiện drift trên 3 kịch bản (recall/FPR); Prometheus scrape; **drift→auto-retrain** kích hoạt đúng | RQ4 · Exp-4 |
| **P7** | CI/CD | C1 | ✅ **MERGED `main`** (local, chưa push) — `ci.yml` (full suite, 3.14), `cd-pipeline.yml` (feature→train→**eval-gate**→deploy), `retrain.yml` (CT trigger), eval-gate module `src/cicd/` (governance candidate→staging, sửa §9.13), DVC `fixture/train/eval` DAG (C2), MLflow server + Prometheus/Grafana compose (smoke-local), §9.10 healthcheck | eval-gate chặn model kém ✓; DVC repro DAG ✓; e2e local smoke (gate→staging→serve→/predict) ✓; DORA/overhead đo ở P8 | RQ3 · maturity |
| **P8a** | Measurement foundation | đánh giá | ✅ **MERGED `main`** — `src/experiments/` (harness N-run subprocess + 6 metric collectors + Wilcoxon/bootstrap CI + **ML Test Score assessor** evidence-auto-verify + provenance §9.6); unit-test trên synthetic, KHÔNG sinh số liệu kết quả | 152 test xanh, ruff sạch; `python -m src.experiments assess` chạy trên manifest thật (mọi điểm credit phải verify được) ✓; collector gắn 1 nhóm metric ✓ | RQ3 · maturity |
| **P8b-1** | RQ3 spine (Exp-1/2/3 + maturity) | đánh giá | ✅ **xong branch `phase8b1`** — Exp-1 (E2E quality real D5: ROC-AUC 0.926), Exp-2 (ablation C0 vs C1, 6 nhóm, Wilcoxon wall p=0.002), Exp-3 (framework Noop/MLflow/ClearML overhead+governance), ML Test Score 28-test (C1=7.0/C0=0.0); chạy thật N=10 → `*_summary.json` + bảng | số thật từ artifact; model-perf/business = control; cost↔benefit định lượng; Wilcoxon N=10 | RQ1,2,3 |
| **P8b-2** | RQ4 + result tables | đánh giá | ⬜ chưa — Exp-4 (drift+retrain), Exp-5 (model-swap), Exp-6 (modifiability) + `raw_data` full run + bảng LaTeX hoàn chỉnh | mỗi Exp ra `*_summary.json`; linh hoạt drift/swap/modify; Wilcoxon N≥10 | RQ4 |
| **P9** | Viết luận văn | — | Migrate LaTeX template; điền kết quả thật; dùng `academic-pipeline` để draft/review | các chương có số liệu thật từ `artifacts/`; không placeholder | — |

## Ánh xạ Experiment → RQ → Phase tạo điều kiện

| Exp | Nội dung | RQ | Cần phase |
|---|---|---|---|
| Exp-1 | E2E + chất lượng phát hiện (ROC-AUC/PR-AUC) | RQ1,RQ2 | P1–P5 |
| Exp-2 | Capability ablation C0 vs C1 (6 metrics) — *xương sống* | RQ2,RQ3 | P4,P7,P8 |
| Exp-3 | Framework MLflow vs ClearML vs Noop (overhead/resource) | RQ3 | P4,P8 |
| Exp-4 | Drift + auto-retrain (sudden/gradual/recurring) | RQ4 | P2,P6 |
| Exp-5 | Model-swap IForest↔LSTM-AE | RQ4 | P4 |
| Exp-6 | Modifiability (đổi feature/rule; module chạm, regression) | RQ4 | P3,P6,P8 |
| Maturity | ML Test Score + Google/Azure (bằng chứng artifact) | RQ3 | P7,P8 |

## Port từ repo cũ (`MLOps_Project/src/`) — dọn sạch khi port

| Logic port lại | Phase |
|---|---|
| `ingestion/parser.py`, `schema.py`, `snapshot.py` | P1 |
| `data/real_profile.py`, `ebs_generator.py`, `synthetic_generator.py`, `generate_datasets.py` | P2 |
| `features/builder.py` (7 feats), `weak_labels.py` | P3 |
| `tracking/__init__.py` → tách `base/noop/mlflow/clearml/factory` | P0/P4 |
| `training/core.py` (IForest), `lstm_detector.py`, `registry.py` | P4 |
| `serving/*` (FastAPI, runtime) | P5 |
| `serving/monitoring.py` (PSI, SystemMetrics, AutoRetrain) | P6 |
| `experiments/*` (harness, records, stats) | P8 |

**Xây mới hoàn toàn:** Feast repo (P3) · DVC pipelines (P1+) · Evidently drift (P6) · GitHub Actions CI/CD + eval-gate (P7) · ML Test Score assessor + capability-ablation runner C0 + Exp-6 modifiability (P8) · pytest suite (mọi phase). **KHÔNG đụng** `nwdaf_mlops/` legacy.

## Mốc kiểm chứng tổng (milestones)

1. **M1 (sau P2):** dữ liệu thật+synthetic versioned, tái lập bằng `dvc repro`.
2. **M2 (sau P4):** train→registry→ "model production" reproducible; C0/C1 + model-swap chạy.
3. **M3 (sau P6):** serving + drift→auto-retrain closed-loop hoạt động trên kịch bản drift.
4. **M4 (sau P7):** CI/CD tự động hóa pipeline + eval-gate → đạt MLOps Level 1(→2).
5. **M5 (sau P8b):** đủ số liệu 6 nhóm metrics + maturity cho cả 4 RQ → sẵn sàng viết. (P8a = hạ tầng đo đã xong; P8b nạp dữ liệu thật.)

> Chi tiết bite-sized từng phase: `docs/plans/<date>-phaseN-*.md` (tạo just-in-time trước khi execute mỗi phase).
