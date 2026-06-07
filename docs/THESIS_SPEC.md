# THESIS DESIGN SPEC — MLOps cho NWDAF (hướng ứng dụng)

> **Trạng thái:** Đã chốt B (thiết kế đối chứng) + C (phạm vi stack) + framework. **Đã self-review (rev.2)** sau khi bỏ ràng buộc thời gian & cho phép build lại từ đầu. Cơ sở học thuật: `docs/research/METHODOLOGY_GROUNDING.md`. Kế hoạch tuần: `docs/IMPLEMENTATION_PLAN.md`.
> **Ngày:** 2026-06-07. **Chủ trương:** không cắt giảm vì thời gian — ưu tiên *hoàn chỉnh, đúng, tái lập*. Build fresh; tái dùng có chọn lọc logic đã kiểm chứng từ repo cũ.

---

## 1. Đề tài & định vị

- **Tên:** *Nghiên cứu và xây dựng quy trình MLOps trong hệ thống phân tích dữ liệu mạng lõi di động thế hệ thứ 5.*
- **Ngành:** Kỹ thuật Phần mềm. **Góc nhìn:** thiết kế & vận hành hệ thống ML (MLOps), KHÔNG phải thuật toán ML mới, KHÔNG phải tối ưu viễn thông.
- **Use case:** phát hiện bất thường **ping-pong handover** của UE từ dữ liệu **EBS** của NWDAF.
- **Luận điểm trung tâm:** *Xây dựng một pipeline MLOps end-to-end, **phủ đủ kiến trúc tham chiếu**, tái lập được, cho phân tích NWDAF; và **đo lường có đối chứng** giá trị MLOps mang lại — bằng metrics khách quan, không chấm tay.*

## 2. Câu hỏi nghiên cứu (đã chốt)

| RQ | Nội dung | Trả lời ở |
|---|---|---|
| **RQ1** | Dữ liệu NWDAF/EBS có đặc thù gì (streaming, nhãn yếu, concept drift, multi-source) và đặt ra thách thức nào cho vận hành ML mà quy trình ad-hoc không đáp ứng? | Ch.2 + Exp-1 |
| **RQ2** | Pipeline MLOps cần thiết kế thế nào (kiến trúc, thành phần, công cụ) để giải quyết các thách thức đó trong bối cảnh NWDAF? | Ch.3 + Exp-1 |
| **RQ3** | MLOps mang lại **giá trị đo được** gì so với quy trình ad-hoc, và framework nào (MLflow/ClearML) phù hợp — xét trên 6 nhóm metrics khách quan? | Ch.4 + Exp-2, Exp-3 |
| **RQ4** | Pipeline có đủ linh hoạt khi dữ liệu trôi dạt (drift→auto-retrain), khi thay mô hình, hoặc khi thay đổi feature/giám sát — mà không phá vỡ quy trình? | Ch.4 + Exp-4, Exp-5, Exp-6 |

## 3. Đóng góp (phát biểu đúng phạm vi — KHÔNG over-claim)

- **C1** — Phân tích hệ thống đặc thù dữ liệu NWDAF/EBS và thách thức vận hành ML (RQ1).
- **C2** — Thiết kế & **hiện thực pipeline MLOps end-to-end phủ ĐỦ 8 thành phần Kreuzberger** (MLflow + DVC + Feast + Evidently + FastAPI + GitHub Actions + Docker) cho bài toán ping-pong handover (RQ2).
- **C3** — Đánh giá **có đối chứng** giá trị MLOps qua *capability ablation* (C0→C1) + *so sánh framework* (MLflow vs ClearML) + *maturity bằng ML Test Score*, trên 6 nhóm metrics đo tự động (RQ3).
- **C4** — Chứng minh tính linh hoạt: drift→auto-retrain, model-swap, và thay đổi feature/giám sát không phá vỡ quy trình (RQ4).

> **Khoảng trống & vị trí:** công trình NWDAF/ML đã công bố (Manias et al. 2022) chủ yếu *thuật toán / phản ứng drift*, **không** đề cập vòng đời MLOps đầy đủ. Đóng góp = **pipeline MLOps vòng đời đầy đủ, tái lập, có đối chứng cho NWDAF** — KHÔNG phải detector mới.

## 4. Thiết kế thực nghiệm (đã chốt)

### 4.1. Hai trục đối chứng
- **Trục 1 — Capability ablation (xương sống, RQ2/RQ3):** cố định *dữ liệu + mô hình + code*, chỉ đổi lớp MLOps. **C0** = pipeline KHÔNG MLOps (script thuần) ; **C1** = full MLOps stack (mục 5). C0 là phiên bản *ablated* của C1 → **không phải straw-man**. Khung theo Google maturity L0→L1(→L2) + "nợ kỹ thuật tránh được" (Sculley 2015). Có thể *ablate từng năng lực* (versioning/tracking/monitoring/CI-CD) để định lượng đóng góp riêng từng phần.
- **Trục 2 — So sánh framework (phụ, RQ3, thỏa thầy Trình):** Noop / **MLflow (chính)** / ClearML ở **lát tracking–registry** (qua `BaseTracker`); ML metrics trùng nhau (sanity), so overhead/resource/governance. Vai trò = *biện minh chọn MLflow*. Kubeflow: xem §8.

### 4.2. Maturity đo khách quan (chống "chấm tay")
- **Chính:** **ML Test Score (Breck et al. 2017)** — 28 test × {0 / 0.5 thủ công / 1 tự động}, điểm = **MIN** 4 mục (Data/Model/Infra/Monitoring), mỗi điểm trỏ **artifact bằng chứng**.
- **Triangulate:** ánh xạ bằng chứng lên Google (L0–2) & Azure (L0–4) dạng present/absent — level *suy ra* từ tiêu chí, không tự khai.
- **ATAM/utility-tree:** chỉ để *chọn* thuộc tính cần đo; KHÔNG báo cáo điểm gán tay làm kết quả.

### 4.3. Danh sách thực nghiệm
| Exp | Mục tiêu | RQ | Cấu hình |
|---|---|---|---|
| **Exp-1** | Pipeline E2E chạy đúng trên EBS handover; chất lượng phát hiện (ROC-AUC/PR-AUC) + baseline timing | RQ1, RQ2 | C1 full |
| **Exp-2** | Capability ablation **C0 vs C1** trên 6 nhóm metrics (giá trị MLOps) | RQ2, RQ3 | C0, C1 |
| **Exp-3** | So sánh framework **Noop/MLflow/ClearML** ở lát tracking (overhead/resource/governance) | RQ3 | C0, MLflow, ClearML |
| **Exp-4** | Drift detection (PSI 3 tầng + KS qua Evidently) + auto-retrain trên kịch bản sudden/gradual/recurring | RQ4 | C1 |
| **Exp-5** | Linh hoạt model-swap (IsolationForest ↔ LSTM-AE) không đổi quy trình | RQ4 | C1 |
| **Exp-6** | **Modifiability:** thêm/đổi feature & thay đổi rule giám sát; đo *khách quan* (số module chạm, số dòng đổi, regression count) — KHÔNG chấm tay | RQ4 | C1 |
| **(xuyên suốt)** | **Maturity** ML Test Score + Google/Azure (bằng chứng artifact) | RQ3 | C0 vs C1 |

### 4.4. Giao thức (đảm bảo tái lập)
N≥10 runs/cấu hình · fixed seeds · loại warmup · **Wilcoxon signed-rank (α=0.05)** · raw JSONL · mean±std + CI95 · cùng train/test split · container hóa.

## 5. Kiến trúc hệ thống — phủ ĐỦ 8 thành phần Kreuzberger

```
EBS (real 3 files + synthetic calibrated)
  → Data pipeline    parse EBS → snapshot → DVC versioning (dvc.yaml DAG)        [C3 + C2]
  → Feature pipeline 7 đặc trưng handover + weak labels → Feast (offline+online) [C3 feature store]
  → Training         IsolationForest / LSTM-AE → MLflow tracking+registry(alias/stage) [C4,C5,C6]
  → Serving          FastAPI /predict (lấy online features từ Feast) + Docker     [C7]
  → Monitoring       PSI + Evidently drift; Prometheus/Grafana ops; auto-retrain  [C8]
  → CI/CD            GitHub Actions: feature → train → eval-gate → serve/deploy    [C1]
  → Tracker layer    BaseTracker → Noop(C0)/MLflow(C1)/ClearML  (biến kiểm soát Exp-2/3)
```

| # | Thành phần Kreuzberger | Công cụ | Ghi chú |
|---|---|---|---|
| C1 | CI/CD | GitHub Actions | build/test/eval-gate/deploy; trigger retrain |
| C2 | Workflow orchestration | **DVC pipelines (`dvc.yaml`) + GitHub Actions** | DAG tái lập (`dvc repro`); nâng cấp orchestrator chuyên dụng nếu cần |
| C3 | Feature store + data versioning | **Feast + DVC** | Feast: online (AnLF real-time) + offline (MTLF batch), point-in-time join; DVC: version dữ liệu EBS |
| C4 | Training infra | sklearn (IForest) / TF (LSTM-AE) + MLflow | tracking trong run |
| C5 | Model registry | MLflow Registry | version + alias + stage (staging→production→archived) |
| C6 | ML metadata store | MLflow Tracking | params/metrics/artifacts/lineage |
| C7 | Model serving | FastAPI + Docker | `/predict`, `/health`, hot-reload, rollback |
| C8 | Monitoring | PSI + Evidently + Prometheus/Grafana | drift (ML) + system (ops) + auto-retrain trigger |

**Nguyên tắc:** backend-agnostic (core qua `BaseTracker`) · mỗi subpackage có `schema.py` + CLI `python -m` · synthetic calibrate từ EBS thực, luôn nêu rõ.

## 6. Bảng truy vết (sợi chỉ đỏ): Thách thức → Thành phần → Công cụ → Metric → RQ

| Thách thức NWDAF | Thành phần | Công cụ | Metric kiểm chứng | RQ |
|---|---|---|---|---|
| Streaming/multi-source EBS | Data pipeline + versioning (C2,C3) | DVC + parser | data-quality %, lineage có/không | RQ1, RQ2 |
| Train/serve feature skew | Feature store (C3) | Feast | feature consistency train↔serve, point-in-time đúng | RQ2 |
| Ít nhãn (weak labels) | Feature/labeling (C4) | rule-based weak label | ROC-AUC, PR-AUC (F1@thr cố định) | RQ1, RQ2 |
| Reproducibility | Metadata + registry (C5,C6) | MLflow | % run pinned+logged, rerun consistency | RQ3 |
| Lựa chọn framework | Tracking layer | MLflow vs ClearML | tracking overhead, RSS, governance | RQ3 |
| Concept drift | Monitoring (C8) | PSI + Evidently | drift recall/FPR, detection latency | RQ4 |
| Model degradation | Feedback/CT (C1,C8) | auto-retrain trigger | recovery time, trigger đúng | RQ4 |
| Linh hoạt thuật toán | Training (C4) | IForest ↔ LSTM-AE | swap không đổi pipeline; Wilcoxon | RQ4 |
| Modifiability (feature/rule) | Pipeline mô-đun | code mô-đun | module chạm, LOC đổi, regression count | RQ4 |
| Vận hành thủ công | CI/CD (C1) | GitHub Actions | DORA: lead time, deploy freq, MTTR | RQ3 |

## 7. Dữ liệu
- **Thực:** 3 file EBS (~233K events, 24K handovers, 8.4K IMSI) → `profile.json`.
- **Synthetic:** generator calibrate từ profile → raw EBS 52 trường + feature parquet; kịch bản drift (sudden/gradual/recurring). **Luôn ghi rõ "synthetic, calibrate từ thực".**
- **7 đặc trưng/(IMSI, cửa sổ 5'):** n_handover, n_unique_cells, pingpong_count, pingpong_rate, mean_inter_ho_s, std_inter_ho_s, entropy_cell_seq. **Weak label:** `pingpong_count≥1 AND n_handover≥3`.

## 8. Phạm vi & quyết định còn mở

**Trong:** pipeline MLOps E2E phủ 8/8 thành phần Kreuzberger cho NWDAF; ping-pong handover; capability ablation C0→C1; MLflow vs ClearML (thực nghiệm); drift+auto-retrain; model-swap; modifiability; **Feast feature store** (đã đưa vào, vì NWDAF có nhu cầu online/offline thật: AnLF real-time vs MTLF batch).
**Ngoài:** thuật toán ML mới; tối ưu viễn thông; "khung đánh giá MLOps tổng quát" (hướng cũ — đã loại).

> ### ✅ QUYẾT ĐỊNH (ĐÃ CHỐT 06/2026 → phương án A) — Kubeflow / Kubernetes
> Vì đã bỏ ràng buộc thời gian, có 2 phương án cho C2 orchestration & "Kubeflow" mà thầy Trình nhắc:
> - **(A) Khuyến nghị — giữ core single-node:** orchestration bằng DVC pipelines + GitHub Actions; Kubeflow xử lý ở mức **literature + hướng phát triển**. → Luận văn **tập trung, hoàn chỉnh, không lệch sang DevOps hạ tầng**. Vẫn phủ đủ 8 thành phần.
> - **(B) Mở rộng — thêm 1 phase Kubernetes:** deploy pipeline lên K8s local (kind/k3s) + **Kubeflow Pipelines / KServe**, biến Kubeflow thành **chiều so sánh framework thực nghiệm thứ 3** (MLflow vs ClearML vs Kubeflow ở mức orchestration/serving). → Mạnh hơn cho RQ3 nhưng tăng độ phức tạp & rủi ro lệch trọng tâm sang hạ tầng.
>
> **ĐÃ CHỐT: (A)** — core single-node; orchestration = DVC pipelines + GitHub Actions; **Kubeflow = literature + hướng phát triển**. (B) chỉ là phần mở rộng *tùy chọn về sau* nếu dư sức, không nằm trong phạm vi chính.

## 9. Threats to validity
| Loại | Threat | Giảm thiểu |
|---|---|---|
| Internal | cùng người thiết kế & đánh giá | metrics tự động, fixed seeds, code + raw JSONL công khai |
| External | 1 use case, synthetic | calibrate từ EBS thực; model-swap; thừa nhận cần dữ liệu nhà mạng thực |
| Construct | ML metrics trùng MLflow/ClearML | đúng thiết kế (sanity); kết luận dựa trên operational/cost/maturity |
| Conclusion | N nhỏ | Wilcoxon, CI, p-value |
| Drift | PSI yếu với gradual | KS/Evidently bổ sung; đề xuất CUSUM/ADWIN hướng phát triển |

## 10. Chiến lược code: build fresh + tái dùng có chọn lọc

Không còn ràng buộc "migrate". **Build mới sạch theo spec này**, tham khảo/port lại *logic đã kiểm chứng* từ `MLOps_Project/src/`:

| Tái dùng (port + dọn sạch) | Xây mới |
|---|---|
| Logic parser EBS, compute 7 features, weak labels, PSI, IForest + LSTM-AE, **mẫu `BaseTracker`** (Noop/MLflow/ClearML), generator synthetic, harness thực nghiệm | **Feast** feature store (online/offline) · **DVC** pipelines + versioning · **Evidently** drift · **GitHub Actions** CI/CD + eval-gate · **ML Test Score** maturity assessor · MLflow registry alias/stage đầy đủ · capability-ablation runner (C0 thật) · Exp-6 modifiability harness |

> KHÔNG dùng `nwdaf_mlops/` legacy. Cấu trúc repo mới theo convention: mỗi subpackage `schema.py` + CLI.
