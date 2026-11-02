document.addEventListener("DOMContentLoaded", () => {
    let currentAnalysisData = null;
    let currentLayer = "overlay"; // overlay | mask | heatmap

    // DOM Elements
    const sampleSelect = document.getElementById("sampleSelect");
    const btnQuick03140 = document.getElementById("btnQuick03140");
    const fileUpload = document.getElementById("fileUpload");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const layerBtns = document.querySelectorAll(".view-layer-btn");

    // Load available sample images from API
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            sampleSelect.innerHTML = '<option value="">-- Chọn ảnh mẫu có Ground Truth --</option>';
            data.samples.forEach(s => {
                const opt = document.createElement("option");
                opt.value = s.id;
                opt.textContent = `Ảnh ID ${s.id} (Ground Truth chuẩn)`;
                sampleSelect.appendChild(opt);
            });
            // Default select 03140 if available
            if (data.samples.some(s => s.id === "03140")) {
                sampleSelect.value = "03140";
                runAnalysis("03140");
            } else if (data.samples.length > 0) {
                sampleSelect.value = data.samples[0].id;
                runAnalysis(data.samples[0].id);
            }
        } catch (err) {
            console.error("Error loading samples:", err);
            sampleSelect.innerHTML = '<option value="03140">Ảnh ID 03140 (Mặc định)</option>';
        }
    }

    // Tab Navigation
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active", "text-emerald-400", "border-emerald-400"));
            navTabs.forEach(t => t.classList.add("text-slate-400", "border-transparent"));
            tab.classList.add("active", "text-emerald-400", "border-emerald-400");
            tab.classList.remove("text-slate-400", "border-transparent");

            const targetId = tab.getAttribute("data-target");
            tabPanes.forEach(p => p.classList.add("hidden"));
            document.getElementById(targetId).classList.remove("hidden");
        });
    });

    // Layer Switcher
    layerBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            layerBtns.forEach(b => {
                b.classList.remove("active", "bg-emerald-500", "text-slate-950");
                b.classList.add("bg-slate-800", "text-slate-300");
            });
            btn.classList.add("active", "bg-emerald-500", "text-slate-950");
            btn.classList.remove("bg-slate-800", "text-slate-300");

            currentLayer = btn.getAttribute("data-layer");
            updateVisualImages();
        });
    });

    // Quick 03140 button
    btnQuick03140.addEventListener("click", () => {
        sampleSelect.value = "03140";
        fileUpload.value = "";
        runAnalysis("03140");
    });

    // Analyze Click
    btnAnalyze.addEventListener("click", () => {
        if (fileUpload.files.length > 0) {
            runUploadAnalysis(fileUpload.files[0]);
        } else if (sampleSelect.value) {
            runAnalysis(sampleSelect.value);
        } else {
            alert("Vui lòng chọn ảnh mẫu hoặc tải ảnh lên!");
        }
    });

    sampleSelect.addEventListener("change", () => {
        if (sampleSelect.value) {
            fileUpload.value = "";
            runAnalysis(sampleSelect.value);
        }
    });

    async function runAnalysis(sampleId) {
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang xử lý...';
        
        try {
            const formData = new FormData();
            formData.append("sample_id", sampleId);
            
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            currentAnalysisData = data;
            renderAll(data);
        } catch (err) {
            alert("Lỗi phân tích: " + err.message);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-play"></i> Phân Tích';
        }
    }

    async function runUploadAnalysis(file) {
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tải lên...';
        
        try {
            const formData = new FormData();
            formData.append("file", file);
            
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            currentAnalysisData = data;
            renderAll(data);
        } catch (err) {
            alert("Lỗi tải ảnh: " + err.message);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-play"></i> Phân Tích';
        }
    }

    function renderAll(data) {
        document.getElementById("badgeSampleId").textContent = "ID: " + data.sample_id;
        document.getElementById("imgInputFace").src = data.face_base64;
        
        const gtBadge = document.getElementById("gtBadge");
        if (data.has_gt) {
            gtBadge.classList.remove("hidden");
            document.getElementById("gtStatusText").textContent = "Có Ground Truth";
            document.getElementById("gtStatusText").className = "text-emerald-400 font-bold";
        } else {
            gtBadge.classList.add("hidden");
            document.getElementById("gtStatusText").textContent = "Ảnh tự do (Không GT)";
            document.getElementById("gtStatusText").className = "text-amber-400";
        }

        // Render predictions images according to current layer
        updateVisualImages();

        // Render Quick Metric badges under cards
        const b = data.models.baseline;
        const a = data.models.attention;
        const s = data.models.dsnake;

        document.getElementById("valBaselineLatency").textContent = b.latency_ms;
        document.getElementById("valAttentionLatency").textContent = a.latency_ms;
        document.getElementById("valDSnakeLatency").textContent = s.latency_ms;

        if (data.has_gt) {
            document.getElementById("valBaselineDice").textContent = b.metrics.dice ? b.metrics.dice.toFixed(3) : "--";
            document.getElementById("valBaselineIoU").textContent = b.metrics.iou ? b.metrics.iou.toFixed(3) : "--";

            document.getElementById("valAttentionDice").textContent = a.metrics.dice ? a.metrics.dice.toFixed(3) : "--";
            document.getElementById("valAttentionIoU").textContent = a.metrics.iou ? a.metrics.iou.toFixed(3) : "--";

            document.getElementById("valDSnakeDice").textContent = s.metrics.dice ? s.metrics.dice.toFixed(3) : "--";
            document.getElementById("valDSnakeIoU").textContent = s.metrics.iou ? s.metrics.iou.toFixed(3) : "--";
        } else {
            document.getElementById("valBaselineDice").textContent = "N/A";
            document.getElementById("valBaselineIoU").textContent = "N/A";
            document.getElementById("valAttentionDice").textContent = "N/A";
            document.getElementById("valAttentionIoU").textContent = "N/A";
            document.getElementById("valDSnakeDice").textContent = "N/A";
            document.getElementById("valDSnakeIoU").textContent = "N/A";
        }

        // Render Tab 2: Morphology (Use DSnake model metrics as gold standard)
        const morph = s.morph;
        document.getElementById("lemperleGradeBadge").textContent = "CẤP " + morph.lemperle_grade;
        document.getElementById("lemperleGradeDesc").textContent = morph.lemperle_desc;
        document.getElementById("morphWAF").textContent = (morph.area_fraction * 100).toFixed(2) + "%";
        document.getElementById("morphLength").textContent = morph.wrinkle_length.toLocaleString() + " px";
        document.getElementById("morphDensity").textContent = morph.density.toFixed(4);
        document.getElementById("morphPixels").textContent = morph.wrinkle_pixels.toLocaleString();
        document.getElementById("imgSkeleton").src = morph.skeleton_base64;

        // Regional Table
        const regTbody = document.getElementById("regionalTableBody");
        regTbody.innerHTML = "";
        morph.regional.forEach(r => {
            const tr = document.createElement("tr");
            let severity = "Bình thường";
            let color = "text-emerald-400";
            if (r.density_pct > 10.0) { severity = "Đậm nét (Độ 3-4)"; color = "text-rose-400"; }
            else if (r.density_pct > 4.0) { severity = "Vừa phải (Độ 2)"; color = "text-amber-400"; }
            else if (r.density_pct > 1.0) { severity = "Nông (Độ 1)"; color = "text-cyan-400"; }

            tr.innerHTML = `
                <td class="py-2.5 px-3 font-medium text-white flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> ${r.zone}
                </td>
                <td class="py-2.5 px-3 font-mono">${r.wrinkle_pixels.toLocaleString()} px</td>
                <td class="py-2.5 px-3 font-mono">${r.density_pct.toFixed(2)}%</td>
                <td class="py-2.5 px-3 font-semibold ${color}">${severity}</td>
            `;
            regTbody.appendChild(tr);
        });

        // Render Tab 3: Model Evaluation Table
        const evalTbody = document.getElementById("evalTableBody");
        evalTbody.innerHTML = "";
        
        const modelsList = [b, a, s];
        modelsList.forEach((m, idx) => {
            const tr = document.createElement("tr");
            const isWinner = (idx === 2); // DSnake
            const highlightClass = isWinner ? "bg-emerald-950/20 text-emerald-300 font-bold" : "";
            
            const diceText = data.has_gt && m.metrics.dice ? m.metrics.dice.toFixed(4) : "Chỉ có với GT";
            const iouText = data.has_gt && m.metrics.iou ? m.metrics.iou.toFixed(4) : "Chỉ có với GT";
            const precText = data.has_gt && m.metrics.precision ? m.metrics.precision.toFixed(4) : "Chỉ có với GT";
            const recText = data.has_gt && m.metrics.recall ? m.metrics.recall.toFixed(4) : "Chỉ có với GT";
            const hd95Text = data.has_gt && m.metrics.hd95 !== undefined ? m.metrics.hd95.toFixed(2) + " px" : "--";
            
            tr.className = highlightClass;
            tr.innerHTML = `
                <td class="py-3 px-4 font-bold flex items-center gap-2">
                    ${isWinner ? '<span class="text-emerald-400"><i class="fa-solid fa-crown text-amber-400"></i> ' + m.name + '</span>' : m.name}
                </td>
                <td class="py-3 px-3 font-mono">${m.params.toLocaleString()}</td>
                <td class="py-3 px-3 font-mono ${isWinner ? 'text-emerald-400 font-extrabold text-sm' : ''}">${diceText}</td>
                <td class="py-3 px-3 font-mono ${isWinner ? 'text-emerald-400 font-extrabold text-sm' : ''}">${iouText}</td>
                <td class="py-3 px-3 font-mono">${precText}</td>
                <td class="py-3 px-3 font-mono">${recText}</td>
                <td class="py-3 px-3 font-mono ${isWinner ? 'text-emerald-400 font-bold' : ''}">${hd95Text}</td>
                <td class="py-3 px-3 font-mono">${m.latency_ms} ms</td>
            `;
            evalTbody.appendChild(tr);
        });
    }

    function updateVisualImages() {
        if (!currentAnalysisData) return;
        const b = currentAnalysisData.models.baseline;
        const a = currentAnalysisData.models.attention;
        const s = currentAnalysisData.models.dsnake;

        let key = "overlay_base64";
        if (currentLayer === "mask") key = "mask_base64";
        if (currentLayer === "heatmap") key = "heatmap_base64";

        document.getElementById("imgBaseline").src = b.visuals[key];
        document.getElementById("imgAttention").src = a.visuals[key];
        document.getElementById("imgDSnake").src = s.visuals[key];
    }

    // Initial load
    loadSamples();
});

