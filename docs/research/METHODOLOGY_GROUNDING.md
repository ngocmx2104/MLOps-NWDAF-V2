# Methodology Grounding — Thiết kế đối chứng & Phạm vi stack

> **Sản phẩm của `/deep-research`** (full mode, 5 luồng điều tra song song, nguồn đã kiểm chứng ở mức primary-source). Dùng để neo Chương Phương pháp & Thực nghiệm của luận văn, và để cập nhật kế hoạch triển khai. Mọi trích dẫn dưới đây đã được verify; nguồn không xác minh được đánh dấu `[UNVERIFIED]` và **không dùng** chính thức.

---

## A. Vấn đề cần giải (từ góp ý hội đồng)

Thầy Trình: (1) chấm điểm chất lượng thủ công/cảm tính → không tái lập; (2) baseline thô → kết luận hiển nhiên; (3) **phải so sánh framework chuẩn**. Phần nghiên cứu dưới đây tìm cơ sở học thuật để thiết kế lại cho đúng.

---

## B. Thiết kế đối chứng — tránh "baseline rơm → kết luận hiển nhiên"

### B.1. Phát hiện cốt lõi: **Capability Ablation** là thiết kế chuẩn mực

Thiết kế phòng vệ tốt nhất trước phê bình "straw-man" là **ablation năng lực với cùng một workload**: cố định *dữ liệu + mô hình + code pipeline*, chỉ **bật/tắt một năng lực MLOps** mỗi lần. Baseline khi đó **chính là pipeline của mình bị gỡ đi một năng lực**, KHÔNG phải một hệ thống yếu do tác giả tự dựng → không thể bị bác là "rigged".

- Điều này khớp trực tiếp với lớp trừu tượng `MLOPS_BACKEND` (C0 noop / C1 ClearML / C2 MLflow) — đây vốn đã là một ablation kinh điển: biến thay đổi *duy nhất* là backend tracking.
- So sánh **framework-vs-framework** (MLflow vs ClearML vs none) là chính đáng *khi workload giống hệt nhau* — so sánh giữa công cụ, không phải giữa "hệ thống tốt" và "baseline bị làm yếu". → thỏa yêu cầu (3) của thầy Trình.

### B.2. Maturity-level progression (trục thứ hai, khách quan)

Khung hóa kết quả như **dịch chuyển trên một thang trưởng thành đã công bố**, thay vì "ad-hoc vs của em":
- **Google MLOps maturity:** Level 0 (manual) → Level 1 (ML pipeline automation, có CT) → Level 2 (CI/CD automation). Mỗi level có tiêu chí *kiểm tra được* (có/không CI, data validation tự động, pipeline trigger…).
- **Azure MLOps maturity:** 5 mức (L0–L4) chấm theo 4 cột People/Model creation/Model release/App integration.
- John, Olsson & Bosch (2021, 2025): maturity model + taxonomy thực nghiệm (14 công ty) — vừa mô tả vừa kê đơn.

### B.3. ATAM / quality attributes — dùng ĐÚNG chỗ

ATAM (Kazman et al. 2000, utility tree + scenario) hợp lệ để **chọn ra** thuộc tính chất lượng cần đo (operability, modifiability, reproducibility). **Điểm yếu (đúng cái thầy Trình chê):** điểm số ATAM là "subjectively appraised", không đặc tả giá trị đo được → **không tái lập**.
→ **Cách dùng phòng vệ:** dùng utility tree chỉ để *biện minh chọn metric nào*; sau đó **gắn mỗi thuộc tính với một metric khách quan, chạy lại được** (overhead ms, drift detection latency, reproducibility error rate, config LOC). **Tuyệt đối không** báo cáo điểm 1–5 gán tay như một kết quả.

### B.4. Khung lý thuyết SE-for-ML để diễn giải số liệu

- **Sculley et al. (2015)** "Hidden Technical Debt in ML Systems": đánh giá hệ ML theo *chi phí bảo trì / nợ kỹ thuật* (entanglement, config debt, undeclared consumers) — đóng khung "MLOps tiết kiệm được nợ gì".
- **Amershi et al. (2019)** quy trình ML 9 giai đoạn (Microsoft) — đóng khung theo maturity quy trình.
- **Lwakatare et al. (2020)**, **Paleyes et al. (2022)**: taxonomy thách thức + đánh giá theo từng giai đoạn vòng đời.

### ✅ Khuyến nghị B (thiết kế đối chứng của luận văn)

Thiết kế **2 trục**, tất cả metric đo tự động:

1. **Trục năng lực (ablation / maturity):** **C0 = pipeline KHÔNG MLOps** (cùng data/model/code, gỡ lớp MLOps) → **C1 = full MLOps stack** (xem mục C). Đo 6 nhóm metrics. Khung kết quả theo Google L0→L1(→L2) và ngôn ngữ "nợ kỹ thuật tránh được" của Sculley. → C0 là phiên bản *ablated*, KHÔNG phải straw-man.
2. **Trục framework (thỏa thầy Trình):** **MLflow (chính) vs ClearML** trên cùng workload; ML metrics phải *trùng nhau* (sanity check), so sánh ở operational/cost/overhead. **Kubeflow = literature** (đòi Kubernetes, ngoài phạm vi single-node — xem C).
3. **Giao thức:** N≥10 runs, fixed seeds, loại warmup, **Wilcoxon signed-rank** (α=0.05), raw JSONL. Báo cáo mean±std + CI.

---

## C. Phạm vi stack — ánh xạ kiến trúc tham chiếu Kreuzberger (8 thành phần)

Kreuzberger, Kühl & Hirschl (2023): 8 thành phần kỹ thuật — **C1** CI/CD · **C2** workflow orchestration · **C3** feature store · **C4** training infra · **C5** model registry · **C6** ML metadata store · **C7** model serving · **C8** monitoring.

### C.1. Ánh xạ công cụ → thành phần

| Công cụ | Thành phần Kreuzberger | Ghi chú |
|---|---|---|
| **MLflow** | **C5 registry + C6 metadata** (chính); một phần C4, C7 | 1 tool phủ C5+C6 — giá trị cốt lõi; có versioning/alias/stage |
| **DVC** | **C3-lite (data versioning) + C2-lite (DAG)** | Bổ trợ MLflow (MLflow không version *dữ liệu*) |
| **Feast** | **C3 feature store** | Online/offline + point-in-time join; tự quản → overhead cao |
| **Evidently AI** | **C8 monitoring** (data/concept drift, data quality) | 100+ metrics; tích hợp MLflow + Grafana |
| **FastAPI** | **C7 serving** | REST endpoint; ghép `prometheus_fastapi_instrumentator` |
| **Docker/Compose** | Cross-cutting (C1, C4, C7) | Nền đóng gói/tái lập |
| **GitHub Actions** | **C1 CI/CD** | Build/test/deploy; trigger retrain khi data commit |
| **Kubeflow** | C2 + C4 + C7 (KServe) | K8s-native; "YAML sprawl"; chỉ đáng ở multi-node/GPU |
| **Prometheus+Grafana** | **C8 monitoring** (system/ops) | Bổ trợ Evidently (ops vs ML-quality) |

### C.2. Trade-off đã dẫn nguồn

- **Feast cho 1 use case / single-node?** Đồng thuận: **overkill** cho đến khi nhiều use case / nhiều team; payoff thật là point-in-time join.
- **Kubeflow?** **Cần cụm Kubernetes** + hàng chục service/CRD + Istio; chỉ đáng ở quy mô lớn → **bỏ, để literature**.
- **DVC vs MLflow artifacts:** DVC chuyên versioning dữ liệu/pipeline kiểu Git; MLflow không version dữ liệu → **bổ trợ, không cạnh tranh**.
- **MLflow:** entry point nhẹ nhất, tích hợp vài dòng, không đổi hạ tầng.

### ✅ Khuyến nghị C (phạm vi cho single-node, ~2 tháng)

**BẮT BUỘC (minimum-viable, phủ C1/C4/C5/C6/C7/C8):**
- **MLflow** (C5+C6) — registry + experiment/metadata tracking.
- **FastAPI** (C7) — serving endpoint.
- **Docker Compose** — tái lập + dựng stack 1 lệnh.
- **GitHub Actions** (C1) — CI/CD cho single repo (miễn phí).
- **Evidently + Prometheus/Grafana** (C8) — drift (ML) + ops metrics. Monitoring là điểm khác biệt cốt lõi của luận văn.

**NÊN CÓ (chi phí thấp, tăng giá trị "reproducibility/lineage" — một bán điểm của luận văn):**
- **DVC** (C3-lite) — data/model versioning EBS → biến "Hướng phát triển" của bản cũ thành "Đã làm".

**BỎ — và biện minh rõ trong luận văn:**
- **Feast** (C3) — overkill cho 1 use case. Nêu lý do bỏ.
- **Kubeflow** (C2/C4) — đòi K8s; chỉ bàn ở phần literature/định hướng.
- **C2 orchestration**: làm nhẹ (`python -m` CLI / Makefile / lightweight orchestrator), không cần engine nặng.

---

## D. Đặc thù miền NWDAF định hình lựa chọn (drift + nhãn yếu)

- **Drift monitoring:** Vì nhãn **thưa** và dữ liệu đến theo **cửa sổ/batch EBS**, các detector *có giám sát* (DDM/EDDM/ADWIN — cần error stream) **bất khả thi lúc suy luận**. → Miền đòi **giám sát phân phối không giám sát: PSI làm tín hiệu 3 tầng chính + KS cho đuôi**, triển khai qua **Evidently**. Gama et al. (2014) đóng khung trade-off **sudden vs gradual** (kịch bản Exp-3). PSI bands 0.1/0.25 là *quy ước thực hành* (credit scoring), không phải ngưỡng thống kê chứng minh.
- **Nhãn yếu (weak supervision):** luật `pingpong_count≥1 AND n_handover≥3` = một *labeling function* kiểu **Snorkel (Ratner et al. 2017)**. Đánh giá dưới nhãn yếu + mất cân bằng: **ROC-AUC + PR-AUC** (Davis & Goadrich 2006; Saito & Rehmsmeier 2015 ưu tiên PR-AUC khi skew); **F1 chỉ ở ngưỡng cố định, có biện minh** (F1 bất ổn theo ngưỡng/prevalence). Nêu counterpoint Movahedi et al. (2024) để chặt chẽ.

---

## E. Khoảng trống nghiên cứu & tuyên bố đóng góp

Công trình NWDAF/ML đã công bố (Manias, Chouman & Shami 2022 — GLOBECOM & MeditCom) **chủ yếu thuật toán / phản ứng drift**; bài MeditCom có drift detection+adaptation nhưng **không** đề cập đầy đủ retraining/registry/versioning/tracking — tức **thiếu vòng đời MLOps end-to-end**.
→ **Đóng góp luận văn KHÔNG phải detector mới**, mà là: **pipeline MLOps vòng đời đầy đủ, tái lập, có đối chứng cho phân tích NWDAF** — gồm lớp tracker backend-agnostic (C0/C1/C2) + drift-triggered retraining — biến một hàm phân tích do 3GPP đặc tả (TS 23.288 Rel-17) thành hệ thống vận hành được, đo được, so sánh được.

---

## F. 6 nhóm metrics — đều đo tự động (chống "chấm tay")

| Nhóm | Metric & cách đo | Nguồn |
|---|---|---|
| **Model performance** | Precision/Recall/F1 (`sklearn`), **ROC-AUC + PR-AUC** (ưu tiên PR-AUC khi skew), MCC | Davis & Goadrich 2006; Saito & Rehmsmeier 2015 |
| **Operational** | pipeline latency/training time (`time.perf_counter`), inference p50/p95/p99 + throughput (Prometheus), **DORA: lead time, deploy freq, change-fail rate, MTTR** | Forsgren/Humble/Kim 2018; Google SRE 2016 |
| **Business impact** *(đo được, không nói suông)* | detections/giờ; **early-detection lead time**; cost-of-missed = FN×C(FN) (≈ signaling overhead/ping-pong, đếm từ EBS); **expected cost** = FP·C(FP)+FN·C(FN) | Elkan 2001 (cost-sensitive) |
| **Drift & data quality** | **PSI** (3 tầng), **KS/Chi-square** (`scipy.stats`), null%/schema/dup rate | Gama 2014; PSI = quy ước thực hành |
| **Cost & resource** | CPU% & **RSS** (`psutil`), peak mem (`tracemalloc`), storage overhead, **tracking overhead** (Δ noop vs MLflow/ClearML = Exp-2) | psutil docs |
| **MLOps maturity** *(khách quan)* | **ML Test Score (Breck et al. 2017): 28 test × {0 / 0.5 manual / 1 automated}, lấy MIN 4 mục**; triangulate Google L0–2 & Azure L0–4 dạng tiêu chí present/absent | Breck et al. 2017; Google; Azure |

> **Maturity & Business Impact** — hai chỗ dễ tái phạm lỗi cảm tính — được "khử chủ quan" bằng: ML Test Score (rubric có sẵn, min-across-sections) và cost-sensitive expected cost (đếm từ EBS). Đây là câu trả lời trực tiếp cho thầy Trình.

---

## G. Tài liệu tham khảo (đã kiểm chứng)

**Kiến trúc & định nghĩa MLOps**
1. Kreuzberger, Kühl, Hirschl (2023). *MLOps: Overview, Definition, and Architecture.* IEEE Access. doi:10.1109/ACCESS.2023.3262138
2. Symeonidis, Nerantzis, Kazakis, Papakostas (2022). *MLOps – Definitions, Tools and Challenges.* IEEE CCWC. arXiv:2201.00162

**Đánh giá / SE-for-ML / maturity**
3. Sculley et al. (2015). *Hidden Technical Debt in ML Systems.* NeurIPS 28.
4. Amershi et al. (2019). *Software Engineering for ML: A Case Study.* ICSE-SEIP. doi:10.1109/ICSE-SEIP.2019.00042
5. Lwakatare et al. (2020). *Large-Scale ML Systems in Real-World Industrial Settings.* Inf. & Softw. Tech. doi:10.1016/j.infsof.2020.106368
6. Paleyes, Urma, Lawrence (2022). *Challenges in Deploying ML: A Survey of Case Studies.* ACM CSUR 55(6). doi:10.1145/3533378
7. Kazman, Klein, Clements (2000). *ATAM.* SEI CMU/SEI-2000-TR-004.
8. John, Olsson, Bosch (2021). *Towards MLOps: A Framework and Maturity Model.* Euromicro SEAA. doi:10.1109/SEAA53835.2021.00050
9. John, Olsson, Bosch (2025). *An Empirical Guide to MLOps Adoption: Framework, Maturity Model and Taxonomy.* Inf. & Softw. Tech. 183:107725. doi:10.1016/j.infsof.2025.107725
10. Breck, Cai, Nielsen, Salib, Sculley (2017). *The ML Test Score.* IEEE Big Data. doi:10.1109/BigData.2017.8258038
11. Hilton et al. (2016). *Usage, Costs, and Benefits of CI in OSS.* ASE. doi:10.1145/2970276.2970358
12. Soares et al. (2022). *Effects of CI: A SLR.* Empirical Softw. Eng. doi:10.1007/s10664-021-10114-1
13. Forsgren, Humble, Kim (2018). *Accelerate (DORA).* IT Revolution Press.

**Drift / weak labels / metrics**
14. Gama, Žliobaitė, Bifet, Pechenizkiy, Bouchachia (2014). *A Survey on Concept Drift Adaptation.* ACM CSUR 46(4):44. doi:10.1145/2523813
15. Bifet & Gavaldà (2007). *ADWIN.* SIAM SDM. doi:10.1137/1.9781611972771.42
16. Gama et al. (2004). *DDM.* SBIA, LNCS 3171.
17. Page (1954). *Continuous Inspection Schemes (CUSUM).* Biometrika 41.
18. Ratner et al. (2017). *Snorkel.* PVLDB 11(3). doi:10.14778/3157794.3157797
19. Davis & Goadrich (2006). *PR vs ROC Curves.* ICML. doi:10.1145/1143844.1143874
20. Saito & Rehmsmeier (2015). *PR plot more informative under imbalance.* PLOS ONE 10(3):e0118432.
21. Elkan (2001). *Foundations of Cost-Sensitive Learning.* IJCAI.

**Miền 5G/NWDAF**
22. 3GPP TS 23.288 (Rel-17). *Architecture enhancements for 5GS to support NWDAF.*
23. Manias, Chouman, Shami (2022). *An NWDAF Approach to 5G Core Signaling Traffic.* IEEE GLOBECOM. arXiv:2209.10428
24. Manias, Chouman, Shami (2022). *A Model Drift Detection and Adaptation Framework for 5G Core.* IEEE MeditCom. arXiv:2209.06852

**`[UNVERIFIED]` — KHÔNG dùng chính thức trước khi xác minh:** arXiv 2602.09292 (SE registered-report template); ResearchGate 281446807 (so sánh ATAM); arXiv 2604.16371 (Systematic Review of MLOps Tools); các paper ping-pong-HO ML lẻ (chỉ trích dẫn theo *chủ đề*). Saito/Movahedi & PSI bands: xem ghi chú trong mục D/F.

---

## H. Tự rà soát (devil's advocate / chất lượng)

- **Phản biện "ablation C0 vẫn là so với không-MLOps":** đã chặn bằng (i) gọi đúng tên *capability ablation* — baseline là chính pipeline bị gỡ năng lực, không phải hệ tự dựng yếu; (ii) bổ sung trục framework MLflow↔ClearML (so sánh công cụ chuẩn, không hiển nhiên); (iii) maturity đo bằng ML Test Score (rubric ngoài, không tự chấm).
- **Nguy cơ over-claim:** chỉ tuyên bố "pipeline MLOps vòng đời đầy đủ cho NWDAF + so sánh có đối chứng", KHÔNG tuyên bố "đánh giá toàn diện".
- **Giới hạn cần ghi rõ trong luận văn:** dữ liệu synthetic calibrate từ EBS thực; single-node; nhãn yếu chưa kiểm chứng bởi chuyên gia; PSI yếu với gradual drift.
