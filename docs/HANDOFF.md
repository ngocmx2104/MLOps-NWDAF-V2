# HANDOFF — Luận văn MLOps NWDAF (bàn giao cho agent session khác)

> **Mục đích:** Bản tổng hợp đầy đủ để một AI agent ở session khác **nắm bối cảnh + bắt tay làm tiếp ngay**. Đọc file này TRƯỚC, rồi đọc `CLAUDE.md` (luật repo, auto-load) + `docs/THESIS_SPEC.md` (spec) + `docs/IMPLEMENTATION_PLAN.md` (roadmap).
> **Cập nhật:** 2026-06-08. **Tiến độ:** 3/9 phase đã merge vào `main`; Phase 4 đang dở (T1–T3 xong, T4–T7 còn lại).

---

## 0. TL;DR cho agent tiếp nhận

- Đây là **luận văn thạc sĩ MLOps cho 5G NWDAF** (phát hiện ping-pong handover), hướng **ứng dụng** — xây pipeline MLOps end-to-end thật + **đo lường có đối chứng** giá trị MLOps. KHÔNG phải "khung đánh giá MLOps tổng quát" (bản cũ bị hội đồng đánh trượt vì điều đó).
- **Đang build fresh** theo 10 phase (P0→P9), **subagent-driven** (mỗi task: implementer → spec review → quality review → fix → merge). Port logic đã kiểm chứng từ repo cũ `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/`.
- **Đã xong & merge `main`:** P0 (scaffold + tracking abstraction), P1 (ingestion + DVC snapshot), P2 (synthetic data + DVC profile/features), P3 (features + **Feast** feature store C3).
- **Đang làm:** P4 (Training + MLflow/ClearML trackers). T1–T3 xong (PyTorch deps, MLflowTracker registry thật, ClearMLTracker). **Việc tiếp theo NGAY:** T4 (port IForest core + training data), T5 (PyTorch LSTM-AE), T6 (training pipeline + CLI), T7 (verify). Plan chi tiết: `docs/plans/2026-06-07-phase4-training-mlflow.md`.
- **54 test xanh, ruff sạch.** Chạy `source .venv/bin/activate && pytest -q && ruff check .` (≈9s).

---

## 1. Bối cảnh & "tại sao như vậy" (đọc kỹ — quyết định mọi thứ)

**Đề tài:** *Nghiên cứu và xây dựng quy trình MLOps trong hệ thống phân tích dữ liệu mạng lõi di động thế hệ thứ 5.* Ngành **Kỹ thuật Phần mềm**, UET-VNU. HV: Mai Xuân Ngọc (24025038). HD: PGS.TS Võ Đình Hiếu.

**Lịch sử "quay xe":**
1. **Bản 1 (PDF/PPTX cũ)** bị hội đồng phê bình nặng (xem `CLAUDE.md` §2):
   - *Thầy Trình:* thao tác hóa thuộc tính chất lượng **thủ công/cảm tính/không tái lập** (vd tự gán `Traceability 0.23→0.92`); baseline thô → kết luận hiển nhiên; claim "đánh giá toàn diện" nói quá; **phải so sánh framework chuẩn (MLflow/Kubeflow)**.
   - *Cô Thủy:* tên đề tài (5G) ≠ nội dung (benchmark MLOps chung).
   - *Thầy Đức Anh:* trình bày không liên quan bài toán.
2. **Bản 2 (hiện tại)** — hướng **ứng dụng**, thầy Hiếu đã duyệt. Không giới hạn thời gian → ưu tiên *hoàn chỉnh, đúng, tái lập*.

**7 anti-patterns TUYỆT ĐỐI tránh** (CLAUDE.md §3): không chấm điểm tay; không baseline rơm rút kết luận hiển nhiên; không biến thành benchmark chung; không claim "toàn diện"; **không bịa số liệu/kết quả**; không đào sâu lý thuyết viễn thông; không dừng ở "đồ án ráp công cụ" (mọi code gắn 1 RQ + 1 metric khách quan).

---

## 2. Quyết định đã KHÓA (không tự ý đổi; hỏi HV nếu cần)

| # | Quyết định | Chi tiết |
|---|---|---|
| **RQ** | 4 câu hỏi | RQ1 đặc thù NWDAF · RQ2 thiết kế pipeline · RQ3 giá trị đo được của MLOps + framework nào · RQ4 linh hoạt (drift/model-swap/modifiability) |
| **B — Thiết kế đối chứng** | 2 trục | **Trục 1 (xương sống): capability ablation C0 (không MLOps) → C1 (full MLOps trên MLflow)**, cùng data/model/code → C0 là bản *ablated*, KHÔNG straw-man. **Trục 2 (phụ): so sánh framework** Noop/MLflow/ClearML ở **lát tracking-registry** (biện minh chọn MLflow). Kubeflow = literature. |
| **C — Stack** | 8/8 thành phần Kreuzberger | MLflow (C5+C6) · DVC (C3 versioning) · **Feast** (C3 feature store) · FastAPI+Docker (C7) · Evidently+Prometheus/Grafana (C8) · GitHub Actions (C1) · C2 = DVC pipelines + Actions |
| **Kubeflow** | Phương án **A** | literature + hướng phát triển; **KHÔNG đưa Kubernetes vào core** (giữ single-node, tập trung) |
| **LSTM-AE** | **PyTorch** (không TensorFlow) | TF chưa có wheel Python 3.14; **torch 2.12 chạy được trên 3.14** → một venv duy nhất. IsolationForest (sklearn) là model chính. |
| **Maturity** | **ML Test Score (Breck 2017)** | 28 test × {0/0.5/1}, lấy MIN 4 mục, trỏ artifact bằng chứng. Triangulate Google L0–2 + Azure L0–4. KHÔNG tự gán điểm. |
| **6 nhóm metrics** | đều đo tự động | Model Performance (ROC-AUC/PR-AUC, F1@ngưỡng cố định) · Operational (DORA, latency) · Business Impact (expected cost = FP·C(FP)+FN·C(FN), đếm từ EBS) · Drift/Data Quality (PSI+KS+Evidently) · Cost/Resource (psutil) · Maturity (ML Test Score) |
| **Drift** | PSI 3 tầng + KS qua **Evidently** | nhãn thưa → KHÔNG dùng detector có giám sát (DDM/ADWIN) |

Cơ sở học thuật (có trích dẫn đã verify): `docs/research/METHODOLOGY_GROUNDING.md`.

---

## 3. Môi trường & repo

- **Repo mới (làm việc):** `/Users/ngocmx/Thạc Sĩ/MLOps_Thesis_Ver2`
- **Repo cũ (nguồn để PORT, đọc-only):** `/Users/ngocmx/Thạc Sĩ/MLOps_Project` — đặc biệt `src/` (ingestion/features/training/tracking/data/experiments/serving). **KHÔNG đụng `nwdaf_mlops/` legacy.**
- **venv:** `.venv` (Python **3.14.2**). Kích hoạt: `source .venv/bin/activate`. Cài: `pip install -e ".[dev,feast,lstm]"`.
- **Deps chính (đã cài, qua `pyproject.toml`):** numpy/pandas/pyarrow/scipy, scikit-learn 1.9, **mlflow**, **dvc[s3]**, **evidently**, fastapi/uvicorn/pydantic, prometheus-client, psutil; extras: `[feast]`=feast 0.63, `[lstm]`=**torch>=2.6** (2.12 đã cài), `[clearml]`=clearml, `[dev]`=pytest/ruff/pre-commit.
- **Git:** branch `main` (P0–P3 merged, HEAD `efc1a9e`) + `phase4-training` (đang làm). Remote `origin` = `git@github.com:ngocmx2104/MLOps-NWDAF-V2.git`. **CHƯA push** — toàn bộ là commit local; HV tự quyết push/merge.
- **Lưu ý Python 3.14:** TensorFlow KHÔNG cài được (dùng PyTorch). Nếu lib nào khác kẹt 3.14 → có sẵn **Python 3.12.8** tại `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`.

---

## 4. Quy trình làm việc (BẮT BUỘC theo)

**Subagent-driven development** (skill `superpowers:subagent-driven-development`). Mỗi phase:
1. **Viết plan chi tiết** `docs/plans/<date>-phaseN-*.md` (bite-sized, TDD, port-ref + code đầy đủ + test code + DVC) — skill `superpowers:writing-plans`.
2. **Tạo branch** `phaseN-<name>` từ `main` (KHÔNG code trên main).
3. **Dispatch implementer subagent** (Agent tool, model sonnet) cho từng nhóm task → **spec-compliance reviewer** → **code-quality reviewer** → fix loop (cùng/khác subagent) → mark done.
4. **Final review** toàn phase → **finishing-a-development-branch**: merge `main` (local fast-forward), verify `pytest`, xóa branch.

**Conventions (giữ nghiêm):**
- **TDD**: viết test fail → implement → pass → commit. **Port "verbatim"** từ repo cũ (code đã dùng `src.` imports → drop-in), chỉ dọn lint (ruff F401/F841/E741) + sửa path khi cần; **ghi rõ deviation**.
- **Commit per task**, message kết bằng dòng trống + trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (Lưu ý: vài commit P3 lỡ ghi "Sonnet 4.6" — cosmetic, đừng sửa history.)
- **Targeted `git add <files>`** — TUYỆT ĐỐI không `git add -A`/`.` (tránh stage `.venv/`, artifacts).
- **KHÔNG `git push`** trừ khi HV yêu cầu. **Merge khi HV chọn** (HV thường gõ "1").
- **Mỗi subpackage `src/<stage>/`** có `schema.py` (config dataclasses) + `cli.py` + `__main__.py`. Mọi logic core đi qua `BaseTracker` (`create_tracker()` đọc env `MLOPS_BACKEND`), không gọi thẳng framework.
- **Artifacts** (`artifacts/`, `data/`, `*.db`, `mlruns/`) gitignored, quản lý bởi DVC; chỉ commit `dvc.yaml`/`dvc.lock`/`*.dvc`.

**Gotchas đã gặp (tránh lặp):**
- **ClearML khởi background threads → pytest không thoát** (trông như treo). Fix: chạy luồng ClearML trong **subprocess + `os._exit(0)` + timeout** (xem `tests/tracking/test_clearml_tracker.py`). ClearML offline: init/log OK; `OutputModel.update_weights` lỗi offline (register_model đã bọc try/except trả id placeholder).
- **MLflow registry** cần backend DB → test dùng `MLFLOW_TRACKING_URI=sqlite:///<tmp>` (không cần server).
- **Lệnh dài tự chuyển nền** (Bash tool) — với pytest nặng/treo, dùng `--ignore`/subprocess/timeout để khoanh vùng, hoặc `TaskStop <id>`.

---

## 5. Tài liệu nguồn (đọc theo thứ tự)

| File | Vai trò |
|---|---|
| `docs/HANDOFF.md` | **File này** — tổng hợp bàn giao |
| `CLAUDE.md` | Luật + bối cảnh + anti-patterns + stack (auto-load mỗi session) |
| `docs/THESIS_SPEC.md` | Spec thiết kế: RQ, đóng góp C1–C4, 6 thực nghiệm, kiến trúc 8 thành phần, **bảng truy vết Thách thức→Thành phần→Công cụ→Metric→RQ** |
| `docs/research/METHODOLOGY_GROUNDING.md` | Deep-research (nguồn đã verify): thiết kế đối chứng, scoping stack, metrics, ML Test Score, gap NWDAF |
| `docs/IMPLEMENTATION_PLAN.md` | Roadmap tổng 10 phase + DoD + mapping Exp→RQ→Phase + danh sách port |
| `docs/plans/2026-06-07-phaseN-*.md` | Plan chi tiết bite-sized từng phase (P0,P1,P2,P3,P4 đã có) |

---

## 6. Roadmap 10 phase + trạng thái

| Phase | Tên | Kreuzberger | Trạng thái |
|---|---|---|---|
| **P0** | Scaffold & Tracking abstraction | nền | ✅ merged (`BaseTracker`/Noop/factory) |
| **P1** | Data ingestion | C2,C3 | ✅ merged (parser/quality/snapshot/CLI + DVC `snapshot`) |
| **P2** | Synthetic data | hỗ trợ | ✅ merged (real_profile/generators + DVC `profile`/`features`; `raw_data` defer) |
| **P3** | Features + **Feast** | C3 | ✅ merged (builder/weak_labels + Feast C3 + DVC `d2_features`/`weak_labels`) |
| **P4** | Training + MLflow/ClearML | C4,C5,C6 | 🟡 **đang làm** — T1–T3 xong, T4–T7 còn lại |
| **P5** | Serving | C7 | ⬜ chưa — FastAPI `/predict` (lấy online feature Feast) + Docker, hot-reload/rollback |
| **P6** | Monitoring + auto-retrain | C8 | ⬜ chưa — PSI + Evidently + Prometheus/Grafana + drift→retrain |
| **P7** | CI/CD | C1 | ⬜ chưa — GitHub Actions (feature→train→eval-gate→deploy) + **docker servers MLflow/ClearML** |
| **P8** | Experiments + Maturity | đánh giá | ⬜ chưa — Exp-1..6 + ML Test Score + Wilcoxon; **chạy full `raw_data` (~60min)** |
| **P9** | Viết luận văn | — | ⬜ chưa — LaTeX, dùng skill `academic-pipeline`; điền số liệu thật |

**Mapping Exp → RQ → Phase cần:** Exp-1 E2E (RQ1,2; P1–P5) · Exp-2 ablation C0vsC1 (RQ2,3; P4,7,8) · Exp-3 framework MLflow/ClearML/Noop (RQ3; P4,8) · Exp-4 drift+retrain (RQ4; P2,6) · Exp-5 model-swap IForest↔LSTM-AE (RQ4; P4) · Exp-6 modifiability (RQ4; P3,6,8) · Maturity ML Test Score xuyên suốt (RQ3; P7,8).

---

## 7. Chi tiết các phase ĐÃ XONG (đã merge `main`)

### P0 — Scaffold & Tracking abstraction (13 test)
- `pyproject.toml` (Python ≥3.11, deps + extras), `.gitignore`, ruff/pytest, pre-commit, `.github/workflows/ci.yml` (lint+test, py3.11, pip cache), `dvc init`.
- **`src/tracking/`**: `schema.py` (`ExperimentConfig`, `RunHandle`), `base.py` (**`BaseTracker` ABC, 7 method**: init_experiment, log_params, log_metrics, log_dataset, **log_artifact**, register_model(+alias), end_experiment), `noop.py` (`NoopTracker` = **C0**), `factory.py` (`create_tracker()` đọc `MLOPS_BACKEND`, lazy-import mlflow/clearml). Interface đã *hardened* (log_artifact, alias, typed dicts).

### P1 — Data ingestion (28 test)
- **`src/ingestion/`**: `schema.py` (52-field EBS positional), `parser.py` (`parse_ebs_files`, `normalize_timestamps`), `quality.py` (8 checks), `snapshot.py` (`build_d1_snapshot`: parse→filter `l_handover`→metadata→D1 parquet+JSON), `cli.py`/`__main__.py`.
- **Dữ liệu thật:** 3 file EBS thật copy vào `data/raw_ebs/` (DVC-tracked). **DVC stage `snapshot`** → D1 từ real EBS.

### P2 — Synthetic data (39 test)
- **`src/data/`**: `real_profile.py` (real EBS → `profile.json`), `synthetic_generator.py` (feature-level + drift, `generate_and_save`), `ebs_generator.py` (raw EBS, 3 behavior × 3 drift + multi-source, `EBSSyntheticGenerator`), `generate_datasets.py` (orchestrator, seeds `[42…6144]`).
- **DVC stages:** `profile` (→ profile.json: **24,347 handovers / 8,393 IMSI / 3,364 cells / 1,534 ping-pong IMSI** — khớp số liệu thật), `features` (13 parquet Dataset E). **`raw_data` ĐỊNH NGHĨA nhưng CHƯA chạy full** (~60min, cho Exp-1/3).

### P3 — Features + Feast (50 test)
- **`src/features/`**: `schema.py` (`WindowConfig`, `WeakLabelConfig`, 7 feature defs), `quality.py` (D2: 5 checks, D5: 4 checks), `builder.py` (`compute_ue_window_features` 7 feature/(IMSI,window), `build_d2_feature_dataset`), `weak_labels.py` (`apply_weak_labels`: `pingpong_count≥1 AND n_handover≥3`), `cli.py`/`__main__.py`.
- **🆕 Feast feature store (C3):** `feast_store.py` (`build_definitions`, `apply_and_materialize`, `get_online_features`, `get_training_features`), `feast_repo/feature_store.yaml` (local + sqlite). **Test train/serve consistency PASS** (so sánh chéo online vs offline store).
- **DVC stages:** `d2_features` (real D2 = **8,393 rows**, QC 5/5), `weak_labels` (D5 positive rate **19.49%**, QC 4/4) — khớp ~19.5% bản cũ.

---

## 8. PHASE 4 đang dở — chi tiết để làm tiếp

**Branch `phase4-training`** (4 commit: `bda1b0d` torch deps, `b5045c8` MLflowTracker, `c1f2dd2` ClearMLTracker, `64abf4d` fix test ClearML; + `abb18df` track plan docs). **54 test xanh.**

**Plan đầy đủ:** `docs/plans/2026-06-07-phase4-training-mlflow.md` (đọc kỹ — có code/test cụ thể từng task).

### ✅ Đã xong (T1–T3)
- **T1:** `pyproject.toml` `[lstm]` = `torch>=2.6`; cập nhật CLAUDE.md §1/§5 + THESIS_SPEC sang PyTorch.
- **T2 — `src/tracking/mlflow_tracker.py`:** `MLflowTracker` khớp interface P0, **registry THẬT** (`mlflow.register_model` + `set_registered_model_alias`). Test dùng sqlite MLflow.
- **T3 — `src/tracking/clearml_tracker.py`:** `ClearMLTracker` khớp interface, `OutputModel` registry. Test = subprocess offline (xem gotcha §4).

### ⬜ CÒN LẠI (T4–T7) — làm tiếp NGAY theo plan
- **T4 — Port training core:** `src/training/__init__.py`, `schema.py` (`TrainingConfig`, `FEATURE_COLUMNS`), `core.py` (`train_isolation_forest`), `data.py` (`apply_sliding_window`, `load_training_dataset`, `prepare_training_matrices`, `split_training_data`) — port verbatim từ `MLOps_Project/src/training/`. + tests.
- **T5 — `src/training/lstm_detector.py` (PyTorch, VIẾT MỚI):** `LSTMAutoencoder` (nn.LSTM encoder→decoder→Linear), `train_lstm_ae`, `predict_lstm_ae`. **Đã có sẵn code đầy đủ trong plan** (đã vá bảo mật `torch.load(weights_only=True)`). + tests.
- **T6 — `src/training/pipeline.py` + `cli.py` + `__main__.py`:** `run_training(dataset_path, model_type, backend, ...)` = **điểm điều khiển C0/C1/C2** (create_tracker → init/log_params/log_metrics/register_model(alias)/end). Test: C0 noop E2E + **model-swap IForest↔LSTM-AE** (RQ4). Code đầy đủ trong plan.
- **T7 — Verify DoD:** `pytest -q && ruff check .` xanh; smoke C1 (MLflow sqlite) qua pipeline; checklist.

**Sau T7:** final review toàn P4 → merge `main` → xóa branch → sang P5.

### Cách bắt đầu T4–T5 (lệnh mẫu cho agent)
Dispatch implementer subagent (sonnet) với prompt: "branch `phase4-training`; đọc `docs/plans/2026-06-07-phase4-training-mlflow.md` Tasks 4–5; port `MLOps_Project/src/training/{schema,core,data}.py` verbatim + viết `lstm_detector.py` PyTorch theo code trong plan; thêm test theo plan; commit per task + trailer; KHÔNG push." → spec review → quality review → fix → tiếp T6 (implementer khác) → T7.

---

## 9. Loose ends / việc nợ (xử lý đúng phase)

1. **`raw_data` DVC stage** (B/C1-C3/D raw EBS, ~60min) bị chạy DỞ ở P2, **chưa lock**. `dvc status` báo "changed". **Trước Exp-1/Exp-3 (P8):** chạy sạch `dvc repro raw_data` rồi commit `dvc.lock`.
2. **`data/models/`** → thêm vào `.gitignore` khi model artifacts xuất hiện (P4 trở đi).
3. **Push lên remote:** chưa push lần nào. Khi HV muốn: `git push` (main + branch). CI GitHub Actions sẽ chạy.
4. **2 commit P3** lỡ dùng trailer "Claude Sonnet 4.6" — cosmetic, không sửa.
5. **MLflow/ClearML servers thật** (docker) cho Exp-2/3 đo overhead → dựng ở **P7** (P4 chỉ sqlite-MLflow + ClearML-offline cho unit test).

---

## 10. BẮT TAY LÀM TIẾP — hành động cụ thể ngay

1. `cd "/Users/ngocmx/Thạc Sĩ/MLOps_Thesis_Ver2" && source .venv/bin/activate && git checkout phase4-training && pytest -q` → xác nhận 54 xanh.
2. Đọc `docs/plans/2026-06-07-phase4-training-mlflow.md` (Tasks 4–7).
3. Thực thi T4→T7 **subagent-driven** (xem §4 + §8). Merge `main` khi HV duyệt.
4. Tiếp P5→P9 theo `docs/IMPLEMENTATION_PLAN.md`: viết plan từng phase → subagent-driven → merge.
5. **P9 viết luận văn:** dùng skill `academic-pipeline`/`academic-paper`; điền **số liệu thật** từ `artifacts/`; bám `vietnamese_thesis_style_guide.md` (port từ repo cũ); LaTeX template port từ `MLOps_Project/Template_thesis_master/`.

---

## 11. Lệnh nhanh hữu ích

```bash
source .venv/bin/activate
pytest -q && ruff check .                 # full suite (≈9s) + lint
pytest tests/<pkg>/test_x.py -v           # 1 file
dvc repro <stage>                         # chạy 1 DVC stage (snapshot/profile/features/d2_features/weak_labels)
dvc status                                # trạng thái pipeline (raw_data sẽ báo "changed" — đã biết)
git log main..HEAD --oneline              # commit của branch hiện tại
MLOPS_BACKEND=noop python -c "from src.tracking import create_tracker; print(type(create_tracker()).__name__)"
```

> **Quy tắc vàng:** mọi con số trong luận văn phải sinh từ code chạy được & tái lập; **KHÔNG BAO GIỜ bịa kết quả**; mọi component gắn 1 RQ + 1 metric khách quan; giữ góc nhìn Kỹ thuật Phần mềm.
