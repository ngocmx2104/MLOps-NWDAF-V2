# THESIS DESIGN SPEC — MLOps cho NWDAF (hướng ứng dụng)

> **Trạng thái:** Đã chốt B (thiết kế đối chứng) + C (phạm vi stack) + framework (06/2026). Đây là spec thiết kế; kế hoạch triển khai theo tuần ở `docs/IMPLEMENTATION_PLAN.md`. Cơ sở học thuật + trích dẫn: `docs/research/METHODOLOGY_GROUNDING.md`.
> **Ngày:** 2026-06-07.

---

## 1. Đề tài & định vị

- **Tên:** *Nghiên cứu và xây dựng quy trình MLOps trong hệ thống phân tích dữ liệu mạng lõi di động thế hệ thứ 5.*
- **Ngành:** Kỹ thuật Phần mềm. **Góc nhìn:** thiết kế & vận hành hệ thống ML (MLOps), KHÔNG phải thuật toán ML mới, KHÔNG phải tối ưu viễn thông.
- **Use case:** phát hiện bất thường **ping-pong handover** của UE từ dữ liệu **EBS** của NWDAF.
- **Luận điểm trung tâm:** *Xây dựng một pipeline MLOps end-to-end, tái lập được, cho phân tích NWDAF; và **đo lường có đối chứng** giá trị mà MLOps mang lại — bằng metrics khách quan, không chấm tay.*

## 2. Câu hỏi nghiên cứu (đã chốt)

| RQ | Nội dung | Trả lời ở |
|---|---|---|
| **RQ1** | Dữ liệu NWDAF/EBS có đặc thù gì (streaming, nhãn yếu, concept drift, multi-source) và đặt ra thách thức nào cho vận hành ML mà quy trình ad-hoc không đáp ứng? | Ch.2 + Exp-1 |
| **RQ2** | Pipeline MLOps cần thiết kế thế nào (kiến trúc, thành phần, công cụ) để giải quyết các thách thức đó trong bối cảnh NWDAF? | Ch.3 + Exp-1 |
| **RQ3** | MLOps mang lại **giá trị đo được** gì so với quy trình ad-hoc, và framework nào (MLflow/ClearML) phù hợp — xét trên 6 nhóm metrics khách quan? | Ch.4 + Exp-2, Exp-3 |
| **RQ4** | Pipeline có đủ linh hoạt khi dữ liệu trôi dạt (drift→auto-retrain) hoặc khi thay mô hình mà không phá vỡ quy trình? | Ch.4 + Exp-4, Exp-5 |

## 3. Đóng góp (phát biểu đúng phạm vi — KHÔNG over-claim)

- **C1** — Phân tích hệ thống đặc thù dữ liệu NWDAF/EBS và thách thức vận hành ML (RQ1).
- **C2** — Thiết kế & **hiện thực** pipeline MLOps end-to-end (ánh xạ 8 thành phần Kreuzberger; stack MLflow + DVC + Evidently + FastAPI + GitHub Actions + Docker) cho bài toán ping-pong handover (RQ2).
- **C3** — Đánh giá **có đối chứng** giá trị MLOps qua *capability ablation* (C0→C1) + *so sánh framework* (MLflow vs ClearML) + *maturity bằng ML Test Score*, trên 6 nhóm metrics đo tự động (RQ3).
- **C4** — Chứng minh tính linh hoạt: drift→auto-retrain và model-swap không phá vỡ quy trình (RQ4).

> **Khoảng trống & vị trí:** công trình NWDAF/ML đã công bố (Manias et al. 2022) chủ yếu *thuật toán / phản ứng drift*, **không** đề cập vòng đời MLOps đầy đủ (tracking, registry, versioning, retraining tự động, so sánh framework). Đóng góp = **pipeline MLOps vòng đời đầy đủ, tái lập, có đối chứng cho NWDAF** — KHÔNG phải detector mới.

## 4. Thiết kế thực nghiệm (đã chốt)

### 4.1. Hai trục đối chứng

- **Trục 1 — Capability ablation (xương sống, RQ2/RQ3):** cố định *dữ liệu + mô hình + code pipeline*, chỉ thay đổi lớp MLOps.
  - **C0** = pipeline KHÔNG MLOps (script thuần: không versioning/tracking/registry/monitoring/CI-CD).
  - **C1** = full MLOps stack (mục 5) trên MLflow.
  - C0 là phiên bản *ablated* của chính C1 → **không phải straw-man**. Khung kết quả theo Google maturity L0→L1(→L2) + ngôn ngữ "nợ kỹ thuật tránh được" (Sculley 2015).
- **Trục 2 — So sánh framework (phụ, RQ3, thỏa thầy Trình):** Noop / **MLflow (chính)** / ClearML ở **lát tracking–registry** (qua `BaseTracker`); ML metrics trùng nhau (sanity), so sánh overhead/resource/governance. Vai trò = *biện minh chọn MLflow*. Kubeflow = literature.

### 4.2. Maturity đo khách quan (chống "chấm tay")

- **Công cụ chính:** **ML Test Score (Breck et al. 2017)** — 28 test × {0 chưa làm / 0.5 thủ công có tài liệu / 1 tự động chạy lặp}, điểm cuối = **MIN** của 4 mục (Data/Model/Infra/Monitoring). Mỗi điểm phải trỏ tới **artifact bằng chứng** (CI job, validation step, drift monitor, retrain trigger).
- **Triangulate:** ánh xạ bằng chứng lên thang Google (L0–2) & Azure (L0–4) dạng tiêu chí present/absent — level được *suy ra* từ tiêu chí thỏa, không tự khai.
- **ATAM/utility-tree:** chỉ dùng để *chọn* thuộc tính cần đo; KHÔNG báo cáo điểm gán tay làm kết quả.

### 4.3. Danh sách thực nghiệm

| Exp | Mục tiêu | RQ | Cấu hình |
|---|---|---|---|
| **Exp-1** | Pipeline E2E chạy đúng trên EBS handover; chất lượng phát hiện (ROC-AUC/PR-AUC) + baseline timing | RQ1, RQ2 | C1 full |
| **Exp-2** | Capability ablation **C0 vs C1** trên 6 nhóm metrics (giá trị MLOps) | RQ2, RQ3 | C0, C1 |
| **Exp-3** | So sánh framework **Noop/MLflow/ClearML** ở lát tracking (overhead/resource/governance) | RQ3 | C0, MLflow, ClearML |
| **Exp-4** | Drift detection (PSI 3 tầng + KS qua Evidently) + auto-retrain trên kịch bản sudden/gradual/recurring | RQ4 | C1 |
| **Exp-5** | Linh hoạt: model-swap (IsolationForest ↔ LSTM-AE) không đổi quy trình | RQ4 | C1 |
| **(xuyên suốt)** | **Maturity** ML Test Score + Google/Azure (bằng chứng artifact) | RQ3 | C0 vs C1 |

### 4.4. Giao thức (đảm bảo tái lập)

N≥10 runs/cấu hình · fixed seeds · loại warmup · **Wilcoxon signed-rank (α=0.05)** cho so sánh cặp · raw results JSONL · báo cáo mean±std + CI95 · cùng train/test split mọi cấu hình · container hóa.

## 5. Kiến trúc hệ thống (ánh xạ 8 thành phần Kreuzberger)

```
EBS (real 3 files + synthetic calibrated)
  → Data pipeline    parse EBS → snapshot → DVC versioning            [C3-lite]
  → Feature pipeline 7 đặc trưng handover/(IMSI,window) + weak labels  [C4 tiền xử lý]
  → Training         IsolationForest (chính) / LSTM-AE (swap) → MLflow tracking+registry (alias/stage)  [C5,C6]
  → Serving          FastAPI /predict + Docker                          [C7]
  → Monitoring       PSI(tự code) + Evidently drift; Prometheus/Grafana ops  [C8]
  → CI/CD            GitHub Actions: feature → train → serve/deploy      [C1]
  → Tracker layer    BaseTracker → Noop(C0)/MLflow(C1)/ClearML  (biến kiểm soát cho Exp-2/3)
```

**Nguyên tắc thiết kế:** backend-agnostic (core đi qua `BaseTracker`, không gọi thẳng API framework) · mỗi subpackage có `schema.py` + CLI `python -m` · synthetic calibrate từ EBS thực, luôn nêu rõ.

## 6. Bảng truy vết (sợi chỉ đỏ): Thách thức → Thành phần → Công cụ → Metric → RQ

| Thách thức NWDAF | Thành phần (Kreuzberger) | Công cụ | Metric kiểm chứng | RQ |
|---|---|---|---|---|
| Streaming/multi-source EBS | Data pipeline + versioning (C3) | DVC + parser | data-quality %, lineage có/không | RQ1, RQ2 |
| Ít nhãn (weak labels) | Feature/labeling (C4) | rule-based weak label | ROC-AUC, PR-AUC (F1@thr cố định) | RQ1, RQ2 |
| Reproducibility | Metadata + registry (C5,C6) | MLflow | % run pinned+logged, rerun consistency | RQ3 |
| Lựa chọn framework | Tracking layer | MLflow vs ClearML | tracking overhead, RSS, governance | RQ3 |
| Concept drift | Monitoring (C8) | PSI + Evidently | drift recall/FPR, detection latency | RQ4 |
| Model degradation | Feedback loop / CT (C1) | auto-retrain trigger | recovery time, retrain trigger đúng | RQ4 |
| Linh hoạt thuật toán | Training (C4) | IForest ↔ LSTM-AE | swap không đổi pipeline; Wilcoxon | RQ4 |
| Vận hành thủ công | CI/CD (C1) | GitHub Actions | DORA: lead time, deploy freq, MTTR | RQ3 |

## 7. Dữ liệu

- **Thực:** 3 file EBS (~233K events, 24K handovers, 8.4K IMSI) → `profile.json` (template thống kê).
- **Synthetic:** generator calibrate từ profile → raw EBS 52 trường + feature-level parquet; kịch bản drift (sudden/gradual/recurring) cho Exp-4. **Luôn ghi rõ "synthetic, calibrate từ thực".**
- **7 đặc trưng/(IMSI, cửa sổ 5'):** n_handover, n_unique_cells, pingpong_count, pingpong_rate, mean_inter_ho_s, std_inter_ho_s, entropy_cell_seq. **Weak label:** `pingpong_count≥1 AND n_handover≥3`.

## 8. Phạm vi

**Trong:** pipeline MLOps E2E cho NWDAF; ping-pong handover; capability ablation C0→C1; MLflow vs ClearML (thực nghiệm), Kubeflow (literature); drift+auto-retrain; model-swap; DVC versioning; CI/CD.
**Ngoài:** thuật toán ML mới; tối ưu viễn thông; Feast feature store; triển khai Kubernetes production; "khung đánh giá MLOps tổng quát" (hướng cũ — đã loại).

## 9. Threats to validity

| Loại | Threat | Giảm thiểu |
|---|---|---|
| Internal | cùng người thiết kế & đánh giá | metrics đo tự động, fixed seeds, code + raw JSONL công khai |
| External | 1 use case, dữ liệu synthetic | calibrate từ EBS thực; chứng minh linh hoạt qua model-swap; thừa nhận cần dữ liệu nhà mạng thực |
| Construct | ML metrics trùng giữa MLflow/ClearML | đúng thiết kế (sanity); kết luận dựa trên operational/cost/maturity |
| Conclusion | N nhỏ | Wilcoxon non-parametric, CI, báo cáo p-value |
| Drift | PSI yếu với gradual | thừa nhận; bổ sung KS/Evidently; đề xuất CUSUM/ADWIN ở hướng phát triển |

## 10. Migrate từ repo cũ (`MLOps_Project/src/`)

| Tái dùng (migrate) | Xây mới / nâng cấp |
|---|---|
| ingestion/parser, features/builder + weak_labels, training/core (IForest) + lstm_detector, serving (FastAPI), monitoring (PSI), **tracking BaseTracker (Noop/MLflow/ClearML)**, data generators, experiments harness | **DVC** versioning · **Evidently** drift (bổ sung cạnh PSI) · **GitHub Actions** CI/CD · **ML Test Score** maturity assessor · MLflow registry alias/stage đầy đủ · capability-ablation runner (C0 thật) |

> KHÔNG copy `nwdaf_mlops/` legacy.
