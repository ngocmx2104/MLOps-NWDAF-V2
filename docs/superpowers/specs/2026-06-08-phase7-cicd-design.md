# Phase 7 — CI/CD (C1) + infra hỗ trợ — DESIGN SPEC

> **Trạng thái:** Đã brainstorm + chốt 4 quyết định nền (Q1–Q4) + 3 refinement sau self-review đối chiếu `THESIS_SPEC.md` (R1–R3). Chờ HV review trước khi sang `writing-plans`.
> **Ngày:** 2026-06-08. **Cơ sở:** `docs/THESIS_SPEC.md` (§5 C1/C2, §6 truy vết, §4.1/§4.3 Exp-2), `docs/IMPLEMENTATION_PLAN.md` (P7 DoD, M4), `docs/HANDOFF.md` §9 (nợ kỹ thuật 9/10/12/13).
> **Phase:** P7 (Kreuzberger **C1 = CI/CD**, hoàn thiện thành phần thứ 8/8). Tiền đề: P0–P6 đã merge `main`, 122 test xanh.

---

## 1. Mục tiêu & academic anchor

**Mục tiêu kỹ thuật:** Tự động hóa pipeline `feature → train → eval-gate → deploy` bằng GitHub Actions; thêm vòng **CT (continuous training) trigger retrain**; dựng **MLflow server thật** (Exp-3) + **observability stack** (Prometheus/Grafana); hoàn thiện DAG orchestration C2 (DVC + Actions).

**Academic anchor (mọi component gắn 1 RQ + 1 metric — chống anti-pattern CLAUDE.md §3.7):**
- C1 trả lời **RQ3** ("MLOps mang giá trị đo được gì so với ad-hoc") qua **Exp-2** (capability ablation C0 vs C1) và **maturity** (ML Test Score + Google L1→L2).
- Metric khách quan: **DORA** (lead time, deploy frequency, MTTR) — `THESIS_SPEC.md` §6 hàng "Vận hành thủ công".
- Phần "trí tuệ" của C1 = **eval-gate** (cổng chất lượng tự động chặn model kém) — năng lực C0 **không** có; là trục chính của ablation Exp-2.

**Không over-claim (CLAUDE.md §3.4):** P7 *xây cơ chế* và *chứng minh nó chạy*; **số liệu DORA/overhead định lượng đo ở P8**, không bịa ở P7 (CLAUDE.md §3.5).

---

## 2. Quyết định phạm vi đã chốt

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| **Q1** | Mô hình thực thi CI/CD | **Hybrid 2 tầng** | CI GitHub-hosted chạy nhẹ + tái lập trên **synthetic fixture nhỏ**; đo lường nặng (DORA real-data, overhead) ở **local/P8**. Tránh phình, không lệ thuộc data thật trên runner, tiết kiệm phút private repo. |
| **Q2** | Server tracking thật | **MLflow thật (nhẹ) + ClearML offline** | MLflow = framework được chọn → đo overhead/RSS/governance đầy đủ. ClearML self-host = 5 container → KHÔNG dựng; ghi nhận độ phức tạp như **finding vận hành** biện minh chọn MLflow (trục phụ). |
| **Q3** | Observability | **Compose Prometheus+Grafana + dashboard, smoke local** | Đúng HANDOFF §9.12: observability KHÔNG sinh metric academic (latency/RSS đã đo trực tiếp) → artifact hoàn thiện C8 + visualize. KHÔNG drift-gauge, KHÔNG daemon loop. |
| **Q4** | Nợ kỹ thuật §9 | **fold §9.10 + §9.13; defer §9.9** | §9.13 = eval-gate governance (lõi C1, bắt buộc). §9.10 = compose model prereq (rẻ, ăn khớp deploy). §9.9 LSTM-registry + torch-Docker → P8 (tránh phình). |
| **R1** | CT/retrain trigger | **Thêm `retrain.yml` + unify eval-gate** | `THESIS_SPEC.md` §5 hàng C1 ghi "trigger retrain"; §6 map CT=(C1,C8) + MTTR. Nối C1→C8, nâng maturity Google L2. Unify gate đồng thời sửa §9.13 tại gốc. |
| **R2** | C2 orchestration | **Thêm stage `train`/`eval` vào `dvc.yaml`; CD chạy `dvc repro`** | `THESIS_SPEC.md` §5 hàng C2 = "DVC pipelines + GitHub Actions", DAG tái lập bằng `dvc repro`. Hiện `dvc.yaml` thiếu train/eval (P4 chưa wire). |
| **R3** | Tài liệu Exp-2/MTTR | **Nêu tường minh C1↔C0 ablation + cơ chế MTTR** | Đảm bảo P8 đo được; bám `THESIS_SPEC.md` §4.1 ("ablate từng năng lực CI-CD"). |

---

## 3. Kiến trúc & ranh giới "auto-test vs smoke thủ công" (xương sống Q1)

| Tầng | Chạy ở đâu | Nội dung | Vai trò luận văn |
|---|---|---|---|
| **CI (auto)** | GitHub Actions | `ruff` + **full pytest** (gồm eval-gate unit tests) | bằng chứng test tự động → ML Test Score (Infra/Monitoring axis) |
| **CD/CT (auto)** | GitHub Actions | `cd-pipeline.yml`: `dvc repro` feature→train→**eval-gate**→deploy-smoke trên **synthetic fixture** (seed cố định), backend **sqlite-MLflow** (exercise governance candidate→staging thật, không cần server) | artifact C1 thật, chứng minh eval-gate + CT end-to-end → maturity L1→L2 |
| **Smoke thủ công (local)** | máy local | MLflow Docker server · Prometheus/Grafana stack · ClearML offline runs | hoàn thiện 8/8 stack (như Docker P5) |
| **Đo lường (P8)** | local | DORA real-data (C0 vs C1) · Exp-3 overhead/RSS | số liệu định lượng — **KHÔNG thuộc P7** |

**Nguyên tắc:** infra nặng + data thật **không nhồi vào CI**. CI = nhẹ/nhanh/tái lập trên fixture.

---

## 4. Deliverables

### A. `.github/workflows/ci.yml` (SỬA — fix bug đã phát hiện)
Bug hiện tại: cài thiếu extras (`pip install -e ".[dev]"` → import feast/torch fail), chạy Python 3.11 ≠ 3.14 local, **chưa từng chạy** (repo chưa push).
**Sửa:** cài `.[dev,feast,lstm,clearml]` (chạy full 122 test), Python **3.14** khớp local (ghi fallback 3.12 nếu runner kẹt wheel), giữ pip cache. Job: `ruff check .` + `pytest`.

### B. `.github/workflows/cd-pipeline.yml` (MỚI — trụ C1)
Trigger: `push` (main) + `workflow_dispatch`. Jobs theo DAG:
1. **feature** — sinh synthetic fixture nhỏ (`synthetic_generator` P2, seed cố định) làm feature input; self-contained, không cần data thật trên runner.
2. **train** — `dvc repro train` (`run_training`, iforest) → model + metrics; backend sqlite-MLflow → đăng ký alias **`candidate`**.
3. **eval-gate** — `python -m src.cicd gate`: kiểm ROC-AUC/PR-AUC ≥ threshold. PASS → promote `candidate`→`staging`. FAIL → exit non-zero, **chặn deploy**.
4. **deploy** — build Docker image + smoke `/health` + `/predict` trên model vừa train (đồng thời giải §9.10). Push GHCR = **optional/defer**.

### C. `src/cicd/` (MỚI — theo convention subpackage)
- `schema.py` — `GateConfig` (thresholds ROC-AUC/PR-AUC, alias names: `candidate`/`staging`/`production`).
- `eval_gate.py` — **module trung tâm** (xem §5): `run_eval_gate(...) -> GateResult`, logic promote candidate→staging.
- `cli.py` / `__main__.py` — `python -m src.cicd gate ...` (workflow gọi).

### D. `src/training/pipeline.py` (SỬA — §9.13, R1)
`run_training` đăng ký alias **`candidate`** (KHÔNG move `staging` trực tiếp). Việc promote `staging` tách ra cho `eval_gate`. Cập nhật test training tương ứng.

### E. `src/monitoring/retrain.py` (SỬA — R1)
`run_retrain_cycle` dùng **chung `src.cicd.eval_gate`** thay cho logic alias inline → sửa §9.13 tại gốc (model retrain bị reject KHÔNG còn để `staging` trỏ nhầm).

### F. `.github/workflows/retrain.yml` (MỚI — R1, CT trigger)
Trigger: `schedule` (cron) + `workflow_dispatch`. Gọi `run_retrain_cycle` (P6) trên fixture/predictions → demo **C1 trigger retrain** (nối C1→C8). Đây là cơ chế CT (Google L2) + nền đo **MTTR**.

### G. MLflow server thật (MỚI — Q2)
`docker-compose.mlflow.yml`: `mlflow server` nhẹ + backend (sqlite/postgres) + artifact volume. Dùng **local** cho Exp-3 (P8). + doc `docs/` ghi nhận **ClearML self-host 5-container = finding vận hành**.

### H. Observability (MỚI — Q3)
- `docker-compose.monitoring.yml`: Prometheus (scrape `/metrics` của P6: `nwdaf_predict_requests_total`, `nwdaf_predict_latency_ms`) + Grafana.
- `prometheus/prometheus.yml` (scrape config) + `grafana/provisioning/` (datasource + **dashboard ops**: latency histogram p50/p95, throughput, request count).
- Drift giữ là `drift_history.jsonl` artifact (§9.12), KHÔNG lên gauge.

### I. DVC DAG (MỚI — R2, C2)
Bổ sung `dvc.yaml`: stage **`train`** (deps: D2 features + weak labels; outs: model + metrics) + stage **`eval`** (deps: model; outs: eval report). Hoàn thiện DAG `dvc repro` cho C2. CD workflow điều phối qua `dvc repro`.
**Nguyên tắc khử nhập nhằng (CI vs local):** stage `train`/`eval` hoàn thiện DAG C2 cho **local/P8** (deps real D2 từ `d2_features`/`weak_labels`). CI **dùng lại cùng code path** trên synthetic fixture nhỏ (seed cố định) — cơ chế chính xác (fixture-stage riêng vs gọi CLI với input parametrized) chốt ở `writing-plans`; **biến khác biệt duy nhất = input dataset**.

### J. §9.10 fix
docker-compose `serving` thêm healthcheck + README/runbook nêu prereq model (hoặc để cd-pipeline tự sinh model trước deploy).

---

## 5. Eval-gate — module trung tâm (1 nguồn sự thật, 3 nơi dùng)

`src/cicd/eval_gate.py` là **single source of truth** cho câu hỏi "model này có được promote không?":

```
                    ┌─────────────────────────┐
 cd-pipeline.yml ──►│                         │
 (model mới)        │   run_eval_gate()       │── PASS ─► promote candidate→staging
 run_retrain_cycle ►│   (ROC-AUC/PR-AUC ≥ thr)│
 (model retrain)    │   + governance          │── FAIL ─► giữ candidate, KHÔNG move staging
 retrain.yml (CT) ──►│                         │            (exit≠0 ở CD → chặn deploy)
                    └─────────────────────────┘
```

**Governance (sửa §9.13):** register (alias `candidate`) **tách khỏi** promote (alias `staging`). Model bị gate từ chối **không bao giờ** chiếm alias `staging`/`production`. Đảm bảo reload/restart sau luôn nạp model đã qua gate.

**Đây là năng lực C0 KHÔNG có** → trục chính ablation Exp-2 (C0 deploy mọi model thủ công, không cổng; C1 có cổng tự động).

---

## 6. Truy vết RQ → component → metric (sợi chỉ đỏ)

| Component | RQ | Metric khách quan (đo ở) |
|---|---|---|
| `ci.yml` full-test | RQ3 | ML Test Score: Infra tests automated (present, P8 assessor) |
| `cd-pipeline.yml` | RQ3 | DORA lead time / deploy freq (P8); maturity L1→L2 |
| `eval_gate` + §9.13 | RQ3 (+RQ4) | gate chặn model kém (unit test, P7); registry correctness |
| `retrain.yml` (CT) | RQ4, RQ3 | trigger đúng; MTTR/recovery (P8) |
| `dvc.yaml` train/eval (C2) | RQ2 | `dvc repro` reproducible (lineage có/không) |
| MLflow server | RQ3 | Exp-3 overhead/RSS (P8) |
| Prometheus/Grafana | RQ3 | C8 completeness (maturity Monitoring axis) |
| §9.10 fix | RQ2 | fresh-clone deploy reproducible (smoke) |

---

## 7. Exp-2 ablation (C0↔C1) & cơ chế MTTR (R3)

- **C1 ablate về C0:** cùng data/model/code; C0 = thao tác `train→deploy` **thủ công bằng script** (không CI, không eval-gate, không CT); C1 = pipeline tự động + eval-gate + CT. **Biến kiểm soát duy nhất = lớp CI/CD.** Metric so sánh: DORA lead-time/deploy-time, số bước thủ công, lỗi do người. (Đo ở P8; P7 đảm bảo cả hai *đo được*.)
- **Cơ chế MTTR** (recovery time DORA) = **eval-gate (chặn model kém) + rollback (P5 `ServingRuntime.rollback`) + retrain-trigger (P7 `retrain.yml`)**. P7 ráp đủ cơ chế; P8 đo thời gian phục hồi trên kịch bản drift.

---

## 8. Chiến lược test (CI-tested vs smoke)

- **CI-tested (auto, GitHub Actions):** `ruff`; full `pytest`; `tests/cicd/test_eval_gate.py` (PASS→promote; FAIL→chặn **và** `staging` KHÔNG trỏ model bị từ chối — governance §9.13); cập nhật test training cho alias `candidate`; `cd-pipeline.yml` chạy trên fixture (feature→train→eval-gate→deploy-smoke); validate workflow YAML.
- **Smoke thủ công (local, như Docker P5):** `docker-compose.mlflow.yml` (server lên, log được); `docker-compose.monitoring.yml` (Prometheus target healthy, Grafana load dashboard); ClearML offline run.

**TDD:** viết test fail → implement → pass → commit (per task). Port logic đã kiểm chứng (eval threshold/metrics từ `MLOps_Project` nếu có) + ghi rõ deviation.

---

## 9. Out-of-scope / defer

- **§9.9** LSTM-registry self-contained + torch-Docker image → **P8**.
- **DORA/overhead real-data** đo định lượng → **P8** (Exp-2/3).
- **`raw_data` full ~60min** (§9.1) → **P8** (Exp-1/3).
- **Push GHCR image** → optional/sau.
- **ClearML self-host stack** → KHÔNG làm (chỉ finding vận hành).
- **Kubeflow/K8s** → literature (đã chốt SPEC §8 phương án A).

---

## 10. Threats to validity (trung thực, bám SPEC §9)

- **ClearML offline:** không đo được server-RSS head-to-head với MLflow (chỉ overhead library-level + finding vận hành 5-container). Ghi rõ ở mục giới hạn — đây là chủ ý phạm vi (trục phụ), không phải sót.
- **CI trên synthetic fixture:** chứng minh *cơ chế* eval-gate/CT chạy, không phải hiệu năng trên data thật (đo ở P8 local). Nêu rõ fixture "synthetic, calibrate từ thực".
- **DORA đơn máy/đơn người:** N nhỏ → P8 dùng Wilcoxon + CI (SPEC §4.4).

---

## 11. File layout (tạo/sửa)

```
.github/workflows/
  ci.yml              (SỬA: full extras, py3.14, ruff+pytest)
  cd-pipeline.yml     (MỚI: dvc repro feature→train→eval-gate→deploy)
  retrain.yml         (MỚI: CT trigger → run_retrain_cycle)
src/cicd/
  __init__.py · schema.py · eval_gate.py · cli.py · __main__.py   (MỚI)
src/training/pipeline.py    (SỬA: alias candidate, không move staging)
src/monitoring/retrain.py   (SỬA: dùng chung src.cicd.eval_gate)
dvc.yaml                    (SỬA: + stage train, eval)
docker-compose.mlflow.yml       (MỚI: MLflow server)
docker-compose.monitoring.yml   (MỚI: Prometheus + Grafana)
prometheus/prometheus.yml       (MỚI: scrape config)
grafana/provisioning/...         (MỚI: datasource + dashboard ops)
docker-compose.yml          (SỬA: healthcheck + §9.10 note)
tests/cicd/test_eval_gate.py    (MỚI)
tests/training/...          (SỬA: alias candidate)
docs/...                    (MỚI: ClearML finding + P7 runbook)
```

---

## 12. Definition of Done (kiểm chứng được — bám IMPLEMENTATION_PLAN P7 + M4)

1. `ci.yml` cài đủ extras + chạy full `pytest` + `ruff` xanh (fix bug 3.11/extras). ✔ khi push CI pass (hoặc `act` local).
2. `cd-pipeline.yml` chạy được end-to-end trên fixture: feature→train→eval-gate→deploy-smoke. ✔ workflow green.
3. **eval-gate chặn model kém** — unit test PASS/FAIL + governance §9.13 (rejected model không chiếm `staging`). ✔ `pytest tests/cicd`.
4. `retrain.yml` gọi `run_retrain_cycle` thành công (CT trigger, C1→C8). ✔ workflow_dispatch run.
5. `dvc.yaml` có stage `train`+`eval`; `dvc repro` chạy DAG (fixture). ✔ `dvc repro` OK.
6. `docker-compose.mlflow.yml` + `docker-compose.monitoring.yml` smoke local OK (server/Prometheus/Grafana lên). ✔ smoke thủ công.
7. §9.10 giải: fresh-clone deploy có hướng dẫn/bootstrap. ✔ README + healthcheck.
8. Toàn bộ test cũ (122) vẫn xanh + test mới; `ruff` sạch.
9. Truy vết RQ/metric (§6) đầy đủ; threats (§10) ghi rõ.

> **Quy tắc vàng (CLAUDE.md):** KHÔNG bịa số liệu — P7 xây cơ chế + chứng minh chạy; số DORA/overhead đo ở P8. Mọi component gắn 1 RQ + 1 metric. Giữ góc nhìn Kỹ thuật Phần mềm, tránh lệch DevOps hạ tầng.
