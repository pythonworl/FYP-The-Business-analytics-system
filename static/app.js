document.addEventListener("DOMContentLoaded", () => {
  console.log("Business Analytics App: Initializing Version 2.0...");

  // UTILS
  function safeInit(name, fn) {
    try {
      fn();
      console.log(`Module loaded: ${name}`);
    } catch (e) {
      console.error(`Module FAILED: ${name}`, e);
    }
  }

  // -----------------------------------------
  // UI ROUTING
  // -----------------------------------------
  const navItems = document.querySelectorAll(".nav-item");
  const panels = document.querySelectorAll(".panel");
  const appCards = document.querySelectorAll(".app-card");

  function switchTab(targetId) {
    navItems.forEach(item => item.classList.remove("active"));
    panels.forEach(panel => panel.classList.remove("active"));

    const activeNav = document.querySelector(`.nav-item[data-target="${targetId}"]`);
    if (activeNav) activeNav.classList.add("active");

    const activePanel = document.getElementById(targetId);
    if (activePanel) {
      activePanel.classList.add("active");
      if (targetId === "report") autoSelectBestReport();
      if (["demand", "sales", "churn"].includes(targetId)) {
          updateModelAvailability();
          refreshAllDropdowns();
      }
      if (targetId === "sales") loadSalesFeatures();
      if (targetId === "churn") loadChurnFeatures();
    }
  }

  navItems.forEach(item => {
    item.addEventListener("click", () => switchTab(item.dataset.target));
  });

  appCards.forEach(card => {
    card.addEventListener("click", () => switchTab(card.dataset.target));
  });

  // -----------------------------------------
  // AUTO ML ENGINE
  // -----------------------------------------
  const btnRunAutoML = document.getElementById("btnRunAutoML");
  const automlStatus = document.getElementById("automlStatus");
  const automlProgressContainer = document.getElementById("automlProgressContainer");
  const automlProgressBar = document.getElementById("automlProgressBar");
  const automlDownload = document.getElementById("automlDownload");

  function clearAutoMLUI() {
    if (automlStatus) automlStatus.innerHTML = "";
    if (automlProgressContainer) automlProgressContainer.style.display = "none";
    if (automlProgressBar) automlProgressBar.style.width = "0%";
    const hub = document.getElementById("modelPerformanceHub");
    if (hub) hub.style.display = "none";
    if (automlDownload) automlDownload.style.display = "none";
  }

  async function fetchAutoMLResults() {
    try {
      const statusRes = await fetch("/api/user_model_status");
      const modelStatus = await statusRes.json();
      
      const res = await fetch("/api/automl_metrics");
      const data = await res.json();
      
      const hub = document.getElementById("modelPerformanceHub");
      const content = document.getElementById("metricsContent");
      const downloadSection = document.getElementById("automlDownload");
      const btnContainer = document.getElementById("downloadButtonsContainer");

      console.log("[DIAGNOSTIC] AutoML State:", { has_preds: data.has_predictions, s: data.has_sales_preds, c: data.has_churn_preds });

      if (data.has_predictions) {
        if (downloadSection) downloadSection.style.display = "block";
        if (automlStatus) automlStatus.innerHTML = `<span style="color:#10b981;">✅ AutoML Cycle Completed. Best models selected!</span>`;
        
        if (btnContainer) {
            btnContainer.innerHTML = "";
            if (data.has_sales_preds) {
                const b = document.createElement("button");
                b.className = "btn";
                b.style.background = "#3b82f6";
                b.innerHTML = "Download Sales Predictions";
                b.onclick = () => { window.location.href = "/api/download/predictions?type=sales"; };
                btnContainer.appendChild(b);
            }
            if (data.has_churn_preds) {
                const b = document.createElement("button");
                b.className = "btn";
                b.style.background = "#8b5cf6";
                b.innerHTML = "Download Churn Predictions";
                b.onclick = () => { window.location.href = "/api/download/predictions?type=churn"; };
                btnContainer.appendChild(b);
            }
        }
      } else {
        if (downloadSection) downloadSection.style.display = "none";
        if (hub) hub.style.display = "none";
      }

      const m = data.metrics;
      if (m && Object.keys(m).length > 0) {
        let html = "";
        let hubVisible = false;
        
        // Error reporting for Sales & Demand
        if (m.sales_demand && m.sales_demand.status === "error") {
            if (automlStatus) automlStatus.innerHTML = `<span style="color:#ef4444;">AutoML Warning: Sales Model Failed - ${m.sales_demand.message}</span>`;
        }
        // Error reporting for Churn
        if (m.churn && m.churn.status === "error") {
            if (automlStatus) automlStatus.innerHTML = `<span style="color:#ef4444;">AutoML Warning: Churn Model Failed - ${m.churn.message}</span>`;
        }

        // ONLY show metrics if the corresponding data file EXISTS and training succeeded
        if (m.sales_demand && m.sales_demand.status === "success" && modelStatus.has_sales_data) {
            hubVisible = true;
            html += `<div style="background:#0f172a; padding:15px; border-radius:8px; border:1px solid #334;">
                <h4 style="margin:0 0 10px; color:#10b981; display:flex; align-items:center; gap:8px;">
                    📈 Sales & Demand Model
                </h4>
                <div style="font-size:0.95rem; margin-bottom:10px;">Champion: <b style="color:#60a5fa; background:rgba(96,165,250,0.1); padding:2px 6px; border-radius:4px;">${m.sales_demand.sales_model}</b></div>
                <div style="display:inline-block; font-size:0.75rem; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3); padding:2px 6px; border-radius:30px; font-weight:600; margin-bottom:10px;">PROVENANCE: Personalized Business Data</div>
                <div style="font-size:0.8rem; margin-top:12px; font-weight:600; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">Algorithm Performance (MAE):</div>
                <ul style="margin:8px 0; padding-left:18px; font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                    ${Object.entries(m.sales_demand.sales_metrics).map(([name, score]) => `<li>${name}: <span style="color:#94a3b8;">${score}</span></li>`).join("")}
                </ul>
            </div>`;
        }
        
        if (m.churn && m.churn.status === "success" && modelStatus.has_churn_data) {
            hubVisible = true;
            html += `<div style="background:#0f172a; padding:15px; border-radius:8px; border:1px solid #334;">
                <h4 style="margin:0 0 10px; color:#10b981; display:flex; align-items:center; gap:8px;">
                    Customer Churn Model
                </h4>
                <div style="font-size:0.95rem; margin-bottom:10px;">Champion: <b style="color:#60a5fa; background:rgba(96,165,250,0.1); padding:2px 6px; border-radius:4px;">${m.churn.churn_model}</b></div>
                <div style="display:inline-block; font-size:0.75rem; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3); padding:2px 6px; border-radius:30px; font-weight:600; margin-bottom:10px;">PROVENANCE: Personalized Business Data</div>
                <div style="font-size:0.8rem; margin-top:12px; font-weight:600; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">Algorithm Performance (F1):</div>
                <ul style="margin:8px 0; padding-left:18px; font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                    ${Object.entries(m.churn.churn_metrics).map(([name, score]) => `<li>${name}: <span style="color:#94a3b8;">${score}</span></li>`).join("")}
                </ul>
            </div>`;
        }
        if (hub) hub.style.display = hubVisible ? "block" : "none";
        if (content) content.innerHTML = html;
      }
    } catch (e) {
      console.warn("Could not fetch metrics", e);
    }
  }

  async function updateModelAvailability() {
    try {
        const res = await fetch("/api/user_model_status");
        const status = await res.json();
        
        const demandBtn = document.getElementById("btnDemand");
        const salesBtn = document.getElementById("btnSales");
        const churnBtn = document.getElementById("btnChurn");

        const updateBtn = (btn, hasModel, type) => {
            if (!btn) return;
            const resDivId = type === 'demand' ? 'demandResult' : (type === 'sales' ? 'salesResult' : 'churnResult');
            const resDiv = document.getElementById(resDivId);
            
            if (hasModel) {
                btn.disabled = false;
                btn.style.opacity = "1";
                btn.title = "Ready to predict";
                if (resDiv && resDiv.innerHTML.includes("Locked")) resDiv.innerHTML = "";
            } else {
                btn.disabled = true;
                btn.style.opacity = "0.5";
                btn.title = `No personalized ${type} model found.`;
                if (resDiv) {
                    resDiv.innerHTML = `<div style="background:rgba(245,158,11,0.1); border:1px solid #f59e0b; padding:15px; border-radius:8px; color:#f59e0b; font-size:0.85rem;">
                        Note - Feature Disabled: Please upload or generate <b>${type}</b> data and run AutoML first to enable this tool.
                    </div>`;
                }
            }
        };

        updateBtn(demandBtn, status.has_qty_model, 'demand');
        updateBtn(salesBtn, status.has_sales_model, 'sales');
        updateBtn(churnBtn, status.has_churn_model, 'churn');
        
    } catch (e) { console.error("Error updating model availability", e); }
  }

  let pollingInterval = null;
  function startPolling() {
    if (!automlProgressContainer || !automlProgressBar || !automlStatus) return;
    automlProgressContainer.style.display = "block";
    
    pollingInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/automl_status");
        const data = await res.json();
        
        if (data.status === "processing") {
          automlProgressBar.style.width = data.progress + "%";
          automlStatus.innerHTML = `<span style="color:#60a5fa;">AutoML in progress... ${data.progress}%</span>`;
        } else if (data.status === "completed") {
          automlProgressBar.style.width = "100%";
          automlStatus.innerHTML = `<span style="color:#10b981;">✅ AutoML Cycle Completed. Best models selected!</span>`;
          clearInterval(pollingInterval);
          fetchAutoMLResults(); 
          updateModelAvailability();
          refreshAllDropdowns();
          loadSalesFeatures();
          loadChurnFeatures();
          setTimeout(() => {
            if (confirm("Automation complete! New insights are ready in the Executive Board. View now?")) {
                const reportNav = document.querySelector('.nav-item[data-target="report"]');
                if (reportNav) reportNav.click();
            }
          }, 800);
        } else if (data.status === "failed") {
          automlStatus.innerHTML = `<span style="color:#ef4444;">AutoML Failed: ${data.error}</span>`;
          clearInterval(pollingInterval);
          if (btnRunAutoML) btnRunAutoML.disabled = false;
        }
      } catch (e) {
        clearInterval(pollingInterval);
      }
    }, 2000);
  }

  safeInit("Home", () => {
    fetchAutoMLResults(); // Load saved metrics and download buttons on page load
    updateModelAvailability();
    btnRunAutoML?.addEventListener("click", async () => {
      try {
        btnRunAutoML.disabled = true;
        const res = await fetch("/api/run_automl", { method: "POST" });
        const data = await res.json();
        if (data.error) {
          automlStatus.innerHTML = `<span style="color:#ef4444;">${data.error}</span>`;
          btnRunAutoML.disabled = false;
        } else {
          startPolling();
        }
      } catch (err) { 
        automlStatus.innerHTML = "Error starting AutoML"; 
        btnRunAutoML.disabled = false;
      }
    });

    document.getElementById("btnDownloadPredictions")?.addEventListener("click", () => {
        window.location.href = "/api/download/predictions";
    });
  });


  // -----------------------------------------
  // UPLOADS (Updated to remove immediate training)
  // -----------------------------------------
  function setupUpload(btnId, fileId, resId, endpoint) {
      const btn = document.getElementById(btnId);
      const fileInp = document.getElementById(fileId);
      const resDiv = document.getElementById(resId);
      if (!btn) return;
      btn.addEventListener("click", async () => {
          if (!fileInp.files.length) { resDiv.innerHTML = '<span style="color:#ef4444;">No file selected.</span>'; return; }
          const formData = new FormData();
          formData.append("file", fileInp.files[0]);
          resDiv.innerHTML = "Uploading...";
          btn.disabled = true;
          try {
              const res = await fetch(endpoint, { method: "POST", body: formData });
              const data = await res.json();
              if (res.ok) {
                  resDiv.innerHTML = `<span style="color:#10b981;">${data.message}</span>`;
                  clearAutoMLUI();
                  updateModelAvailability();
                  refreshAllDropdowns();
                  loadSalesFeatures();
                  loadChurnFeatures();
              } else {
                  resDiv.innerHTML = `<span style="color:#ef4444;">${data.error}</span>`;
              }
          } catch (err) { resDiv.innerHTML = "Upload failed."; }
          finally { btn.disabled = false; }
      });
  }
  setupUpload("btnUploadSales", "fileSalesData", "salesUploadResult", "/api/upload/sales");
  setupUpload("btnUploadChurn", "fileChurnData", "churnUploadResult", "/api/upload/churn");
  setupUpload("btnUploadReport", "fileReportData", "reportUploadResult", "/api/upload/report");


  // -----------------------------------------
  // INTELLIGENCE CHAT
  // -----------------------------------------
  const chatInput = document.getElementById("chatInput");
  const btnChatSend = document.getElementById("btnChatSend");
  const chatHistory = document.getElementById("chatHistory");

  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    appendMessage(text, "user");
    chatInput.value = "";
    const loadingId = "bot-loading-" + Date.now();
    appendMessage("thinking...", "bot", loadingId);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      document.getElementById(loadingId)?.remove();
      if (data.error) appendMessage("Error: " + data.error, "bot");
      else appendMessage(data.reply, "bot");
    } catch (err) {
      document.getElementById(loadingId)?.remove();
      appendMessage("Connection error.", "bot");
    }
  }

  function appendMessage(text, type, id = null) {
    const div = document.createElement("div");
    div.className = type === "user" ? "user-msg" : "bot-msg";
    if (id) div.id = id;
    if (type === "bot") {
      let formatted = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
      div.innerHTML = formatted;
    } else {
      div.textContent = text;
    }
    if (chatHistory) {
      chatHistory.appendChild(div);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  }

  safeInit("Chat", () => {
    if (btnChatSend && chatInput) {
      btnChatSend.addEventListener("click", sendChatMessage);
      chatInput.addEventListener("keypress", (e) => { if (e.key === "Enter") sendChatMessage(); });
    }
  });

  // -----------------------------------------
  // DROPDOWNS & PREDICTIONS
  // -----------------------------------------
  const dCategory = document.getElementById("d_category");
  const dSubcategory = document.getElementById("d_subcategory");
  const sCategory = document.getElementById("s_category");
  const sSubcategory = document.getElementById("s_subcategory");

  async function refreshAllDropdowns() {
    try {
      const res = await fetch("/api/options/all");
      const d = await res.json();
      
      const updateSelect = (id, vals) => {
          const el = document.getElementById(id);
          if (!el) return;
          const current = el.value;
          el.innerHTML = vals.map(v => `<option value="${v}">${v}</option>`).join("");
          if (vals.includes(current)) el.value = current;
      };

      updateSelect("d_category", d.categories);
      updateSelect("s_category", d.categories);
      updateSelect("f_category", ["All", ...d.categories]);
      updateSelect("d_region", d.regions);
      updateSelect("s_region", d.regions);
      updateSelect("d_year", d.years);
      updateSelect("s_year", d.years);
      
      // Refresh subcategories for whatever the new category is
      loadSubcategoriesForCategory(document.getElementById("d_category").value, document.getElementById("d_subcategory"));
      loadSubcategoriesForCategory(document.getElementById("s_category").value, document.getElementById("s_subcategory"));
      
    } catch (e) { console.error("Error refreshing dropdowns", e); }
  }

  async function loadSubcategoriesForCategory(category, targetSelect) {
    if (!targetSelect || !category) return;
    try {
      const res = await fetch(`/api/options/subcategories?category=${encodeURIComponent(category)}`);
      const data = await res.json();
      const subs = data.subcategories || [];
      targetSelect.innerHTML = "";
      if (subs.length === 0) {
        targetSelect.innerHTML = '<option value="">None</option>';
        targetSelect.disabled = true;
      } else {
        targetSelect.disabled = false;
        subs.forEach(s => {
          const opt = document.createElement("option");
          opt.value = s; opt.textContent = s;
          targetSelect.appendChild(opt);
        });
      }
    } catch (err) { console.error(err); }
  }

  safeInit("Demand", () => {
    if (dCategory && dSubcategory) {
      dCategory.addEventListener("change", () => loadSubcategoriesForCategory(dCategory.value, dSubcategory));
      loadSubcategoriesForCategory(dCategory.value, dSubcategory);
    }
    
    const triggerDemand = () => { document.getElementById("btnDemand")?.click(); };
    ["d_category", "d_subcategory", "d_region", "d_month"].forEach(id => {
      document.getElementById(id)?.addEventListener("change", triggerDemand);
    });

    document.getElementById("btnDemand")?.addEventListener("click", async () => {
      const resDiv = document.getElementById("demandResult");
      const payload = {
        category: dCategory.value, sub_category: dSubcategory.value,
        region: document.getElementById("d_region").value,
        year: document.getElementById("d_year").value, month: document.getElementById("d_month").value,
      };
      resDiv.innerHTML = "Predicting...";
      try {
        const res = await fetch("/api/predict/demand", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const out = await res.json();
        
        if (!res.ok || out.error) {
            resDiv.innerHTML = `<div style="background:rgba(239,68,68,0.1); border:1px solid #ef4444; padding:15px; border-radius:8px; color:#ef4444; font-size:0.9rem;">
                <b>Prediction Error:</b> ${out.error || "Unknown server error"}
            </div>`;
            return;
        }

        const badge = `<div style="display:inline-block; font-size:0.7rem; background:rgba(96,165,250,0.1); color:#60a5fa; border:1px solid rgba(96,165,250,0.3); padding:2px 6px; border-radius:30px; margin-top:10px;">Model: ${out.model_source}</div>`;
        resDiv.innerHTML = `<div class="card"><div class="big">Predicted Demand: <b>${out.predicted_total_quantity}</b></div><div class="muted">Stats factor: ${out.stats_mode}</div>${badge}</div>`;
      } catch (err) { 
          resDiv.innerHTML = `<div style="color:#ef4444;">Network error: Could not reach prediction server.</div>`; 
      }
    });
  });

  // -----------------------------------------
  // SALES — DYNAMIC FEATURE RENDERING
  // -----------------------------------------
  let salesFeaturesMeta = null;

  function renderSalesInputs(metadata) {
    const container = document.getElementById("salesDynamicInputs");
    const featureList = document.getElementById("salesFeatureList");
    const featureStats = document.getElementById("salesFeatureStats");
    if (!container) return;

    if (!metadata || !metadata.has_features || !metadata.features || metadata.features.length === 0) {
      container.innerHTML = `<div style="grid-column: 1 / -1; text-align:center; padding:30px; color:var(--text-muted); font-size:0.9rem;">
        <div style="font-size:2rem; margin-bottom:10px;"></div>
        <div>No sales model trained yet.</div>
        <div style="font-size:0.8rem; margin-top:5px; color:#94a3b8;">Upload or generate sales data and run AutoML — the system will automatically detect the most important features from your data.</div>
      </div>`;
      if (featureList) featureList.innerHTML = '<div style="padding:10px 0; color:#94a3b8;">Run AutoML to discover features from your data.</div>';
      if (featureStats) featureStats.style.display = "none";
      return;
    }

    salesFeaturesMeta = metadata;
    let inputsHTML = '';
    let featureTagsHTML = '';

    // Filter out auto-derived date features from user input (they are computed automatically)
    const autoFeatures = ['Order_Month', 'Order_Quarter', 'Order_Year'];

    metadata.features.forEach((feat) => {
      const badge = getImportanceBadge(feat.importance);
      const humanName = humanizeFeatureName(feat.name);
      const impPercent = (feat.importance * 100).toFixed(1);
      const fieldId = `sales_dyn_${feat.name}`;
      const isAutoFeature = autoFeatures.includes(feat.name);

      const impBadge = `<span style="display:inline-block; font-size:0.65rem; background:${badge.bg}; color:${badge.color}; border:1px solid ${badge.border}; padding:1px 5px; border-radius:20px; margin-left:6px; font-weight:600;">${badge.label}</span>`;

      if (!isAutoFeature) {
        if (feat.type === "numeric") {
          const step = (feat.max - feat.min) > 100 ? "1" : "0.01";
          const defaultVal = feat.median !== undefined ? feat.median : "";
          inputsHTML += `<label>${humanName} ${impBadge}
            <input id="${fieldId}" type="number" step="${step}" value="${defaultVal}" 
                   min="${feat.min}" max="${feat.max * 2}" 
                   placeholder="${feat.min} – ${feat.max}"
                   data-feature="${feat.name}" data-type="numeric" />
          </label>`;
        } else if (feat.type === "categorical") {
          let optsHTML = `<option value="">Select...</option>`;
          (feat.values || []).forEach(v => {
            optsHTML += `<option value="${v}">${v}</option>`;
          });
          inputsHTML += `<label>${humanName} ${impBadge}
            <select id="${fieldId}" data-feature="${feat.name}" data-type="categorical">
              ${optsHTML}
            </select>
          </label>`;
        }
      }

      // Sidebar feature tags
      const autoLabel = isAutoFeature ? ' <span style="font-size:0.6rem; color:#94a3b8;">(auto)</span>' : '';
      featureTagsHTML += `<div class="requirement-tag" style="border-left:3px solid ${badge.color}; margin:4px 0; padding:3px 8px; display:flex; justify-content:space-between; align-items:center;">
        <span>${feat.name}${autoLabel}</span>
        <span style="font-size:0.7rem; color:${badge.color}; font-weight:700;">${impPercent}%</span>
      </div>`;
    });

    container.innerHTML = inputsHTML;

    if (featureList) featureList.innerHTML = featureTagsHTML;
    if (featureStats) {
      featureStats.style.display = "block";
      featureStats.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span>Total Features:</span><span style="color:#60a5fa; font-weight:600;">${metadata.total_features}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span>Training Samples:</span><span style="color:#60a5fa; font-weight:600;">${(metadata.total_samples || 0).toLocaleString()}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span>Target:</span><span style="color:#a78bfa; font-weight:600;">${metadata.target_column || 'Sales'}</span>
        </div>
      `;
    }
  }

  async function loadSalesFeatures() {
    try {
      const res = await fetch("/api/sales/features");
      const data = await res.json();
      renderSalesInputs(data);
    } catch (e) {
      console.warn("Could not load sales features:", e);
    }
  }

  // Sales Prediction Handler (Dynamic)
  document.getElementById("btnSales")?.addEventListener("click", async () => {
    const resDiv = document.getElementById("salesResult");

    // Collect values from dynamic inputs
    const payload = {};
    const dynamicInputs = document.querySelectorAll("#salesDynamicInputs [data-feature]");

    if (dynamicInputs.length === 0) {
      resDiv.innerHTML = `<div style="background:rgba(245,158,11,0.1); border:1px solid #f59e0b; padding:15px; border-radius:8px; color:#f59e0b; font-size:0.9rem;">
        No features loaded. Please generate or upload sales data and run AutoML first.
      </div>`;
      return;
    }

    dynamicInputs.forEach(el => {
      const featureName = el.dataset.feature;
      payload[featureName] = el.value;
    });

    resDiv.innerHTML = "Predicting...";
    try {
      const res = await fetch("/api/predict/sales", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const out = await res.json();

      if (!res.ok || out.error) {
          resDiv.innerHTML = `<div style="background:rgba(239,68,68,0.1); border:1px solid #ef4444; padding:15px; border-radius:8px; color:#ef4444; font-size:0.9rem;">
              <b>Prediction Error:</b> ${out.error || "Unknown server error"}
          </div>`;
          return;
      }

      const badge = `<div style="display:inline-block; font-size:0.7rem; background:rgba(96,165,250,0.1); color:#60a5fa; border:1px solid rgba(96,165,250,0.3); padding:2px 6px; border-radius:30px; margin-top:10px;">Model: ${out.model_source}</div>`;
      resDiv.innerHTML = `<div class="card"><div class="big">Predicted Revenue: <b>${out.predicted_sales}</b></div>${badge}</div>`;
    } catch (err) { 
        resDiv.innerHTML = `<div style="color:#ef4444;">Network error: Could not reach prediction server.</div>`; 
    }
  });

  // -----------------------------------------
  // FORECASTING
  // -----------------------------------------
  let forecastChart = null;

  document.getElementById("btnForecast")?.addEventListener("click", async () => {
      const resDiv = document.getElementById("forecastResult");
      const metricsDiv = document.getElementById("forecastMetrics");
      const ctx = document.getElementById("forecastChart")?.getContext("2d");
      
      metricsDiv.innerHTML = "Generating forecast...";
      try {
          const resp = await fetch("/api/forecast/sales_series", { 
              method: "POST", 
              headers: { "Content-Type": "application/json" }, 
              body: JSON.stringify({ 
                  category: document.getElementById("f_category").value, 
                  horizon: document.getElementById("f_horizon").value 
              }) 
          });
          const data = await resp.json();
          
          if (!resp.ok || data.error) {
              metricsDiv.innerHTML = `<span style="color:#ef4444;">Forecast Error: ${data.error || "Unknown error"}</span>`;
              if (forecastChart) forecastChart.destroy();
              return;
          }

          if (forecastChart) forecastChart.destroy();
          const allLabels = [...data.history.dates, ...data.forecast.dates];
          const plotHistory = [...data.history.values, ...new Array(data.forecast.dates.length).fill(null)];
          const plotForecast = [...new Array(data.history.dates.length - 1).fill(null), data.history.values[data.history.values.length - 1], ...data.forecast.values];
          
          forecastChart = new Chart(ctx, { 
              type: 'line', 
              data: { 
                  labels: allLabels, 
                  datasets: [
                      { 
                          label: 'Historical Sales', 
                          data: plotHistory, 
                          borderColor: '#60a5fa',         // Brighter Blue
                          backgroundColor: 'rgba(96, 165, 250, 0.15)', 
                          fill: true, 
                          tension: 0.4, 
                          pointRadius: 6, 
                          pointHoverRadius: 8,
                          borderWidth: 4                  // Thicker Line
                      }, 
                      { 
                          label: 'AI Forecast Trend', 
                          data: plotForecast, 
                          borderColor: '#34d399',         // Brighter Green
                          backgroundColor: 'rgba(52, 211, 153, 0.1)', 
                          borderDash: [8, 4], 
                          fill: true,                     // Shine under forecast too
                          tension: 0.4, 
                          pointRadius: 6, 
                          pointHoverRadius: 8,
                          borderWidth: 4                  // Thicker Line
                      }
                  ] 
              },
              options: {
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: { intersect: false, mode: 'index' },
                  plugins: { 
                      legend: { display: true, labels: { color: '#f8fafc', font: { size: 14, weight: 'bold' } } },
                      tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)', titleColor: '#f8fafc', bodyColor: '#cbd5e1', padding: 12 } 
                  },
                  scales: {
                      x: { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8', font: { size: 12 } } },
                      y: { 
                          grid: { color: 'rgba(255,255,255,0.08)' }, 
                          ticks: { color: '#94a3b8', font: { size: 12 } }, 
                          beginAtZero: false,             // Zoom in on the data range
                          grace: '10%'                    // Add some padding at top/bottom
                      }
                  }
              }
          });
          metricsDiv.innerHTML = `Model: <b>${data.model_name}</b> | Error (MAE): <b>${data.mae}</b> | Frequency: <b>${data.freq}</b>`;
      } catch (err) { 
          metricsDiv.innerHTML = `<span style="color:#ef4444;">Network error: Could not reach forecasting engine.</span>`; 
      }
  });

  // -----------------------------------------
  // CHURN — DYNAMIC FEATURE RENDERING
  // -----------------------------------------
  let churnFeaturesMeta = null;  // Cached feature metadata

  function getImportanceBadge(importance) {
    if (importance >= 0.20) return { label: "🔥 High Impact", color: "#10b981", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.3)" };
    if (importance >= 0.10) return { label: "⚡ Medium", color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.3)" };
    return { label: "○ Low", color: "#64748b", bg: "rgba(100,116,139,0.1)", border: "rgba(100,116,139,0.3)" };
  }

  function humanizeFeatureName(name) {
    // Convert snake_case or CamelCase to readable labels
    return name
      .replace(/([A-Z])/g, ' $1')       // CamelCase → spaces
      .replace(/_/g, ' ')                // snake_case → spaces
      .replace(/\b\w/g, c => c.toUpperCase())  // Capitalize words
      .trim();
  }

  function renderChurnInputs(metadata) {
    const container = document.getElementById("churnDynamicInputs");
    const featureList = document.getElementById("churnFeatureList");
    const featureStats = document.getElementById("churnFeatureStats");
    if (!container) return;

    if (!metadata || !metadata.has_features || !metadata.features || metadata.features.length === 0) {
      container.innerHTML = `<div style="grid-column: 1 / -1; text-align:center; padding:30px; color:var(--text-muted); font-size:0.9rem;">
        <div style="font-size:2rem; margin-bottom:10px;"></div>
        <div>No churn model trained yet.</div>
        <div style="font-size:0.8rem; margin-top:5px; color:#94a3b8;">Upload or generate churn data and run AutoML — the system will automatically detect the most important features from your data.</div>
      </div>`;
      if (featureList) featureList.innerHTML = '<div style="padding:10px 0; color:#94a3b8;">Run AutoML to discover features from your data.</div>';
      if (featureStats) featureStats.style.display = "none";
      return;
    }

    churnFeaturesMeta = metadata;
    let inputsHTML = '';
    let featureTagsHTML = '';

    metadata.features.forEach((feat, idx) => {
      const badge = getImportanceBadge(feat.importance);
      const humanName = humanizeFeatureName(feat.name);
      const impPercent = (feat.importance * 100).toFixed(1);
      const fieldId = `churn_dyn_${feat.name}`;

      // Build the importance indicator
      const impBadge = `<span style="display:inline-block; font-size:0.65rem; background:${badge.bg}; color:${badge.color}; border:1px solid ${badge.border}; padding:1px 5px; border-radius:20px; margin-left:6px; font-weight:600;">${badge.label}</span>`;

      if (feat.type === "numeric") {
        const step = (feat.max - feat.min) > 100 ? "1" : "0.1";
        const defaultVal = feat.median !== undefined ? feat.median : "";
        inputsHTML += `<label>${humanName} ${impBadge}
          <input id="${fieldId}" type="number" step="${step}" value="${defaultVal}" 
                 min="${feat.min}" max="${feat.max}" 
                 placeholder="${feat.min} – ${feat.max}"
                 data-feature="${feat.name}" data-type="numeric" />
        </label>`;
      } else if (feat.type === "categorical") {
        let optsHTML = `<option value="">Select...</option>`;
        (feat.values || []).forEach(v => {
          optsHTML += `<option value="${v}">${v}</option>`;
        });
        inputsHTML += `<label>${humanName} ${impBadge}
          <select id="${fieldId}" data-feature="${feat.name}" data-type="categorical">
            ${optsHTML}
          </select>
        </label>`;
      }

      // Sidebar feature tags (color-coded by importance)
      featureTagsHTML += `<div class="requirement-tag" style="border-left:3px solid ${badge.color}; margin:4px 0; padding:3px 8px; display:flex; justify-content:space-between; align-items:center;">
        <span>${feat.name}</span>
        <span style="font-size:0.7rem; color:${badge.color}; font-weight:700;">${impPercent}%</span>
      </div>`;
    });

    container.innerHTML = inputsHTML;

    // Update sidebar
    if (featureList) {
      featureList.innerHTML = featureTagsHTML;
    }
    if (featureStats) {
      featureStats.style.display = "block";
      const labelInfo = metadata.is_churn_inverted 
        ? `<span style="color:#f59e0b;">1 = Churned</span>` 
        : `<span style="color:#10b981;">1 = Purchased</span>`;
      featureStats.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span>Total Features:</span><span style="color:#60a5fa; font-weight:600;">${metadata.total_features}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span>Training Samples:</span><span style="color:#60a5fa; font-weight:600;">${(metadata.total_samples || 0).toLocaleString()}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span>Target:</span><span style="color:#a78bfa; font-weight:600;">${metadata.original_target_name || metadata.target_column || 'PurchaseStatus'}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span>Label:</span>${labelInfo}
        </div>
      `;
    }
  }

  async function loadChurnFeatures() {
    try {
      const res = await fetch("/api/churn/features");
      const data = await res.json();
      renderChurnInputs(data);
    } catch (e) {
      console.warn("Could not load churn features:", e);
    }
  }

  // Churn Prediction Handler (Dynamic)
  document.getElementById("btnChurn")?.addEventListener("click", async () => {
    const resDiv = document.getElementById("churnResult");

    // Collect values from dynamic inputs
    const payload = {};
    const dynamicInputs = document.querySelectorAll("#churnDynamicInputs [data-feature]");

    if (dynamicInputs.length === 0) {
      resDiv.innerHTML = `<div style="background:rgba(245,158,11,0.1); border:1px solid #f59e0b; padding:15px; border-radius:8px; color:#f59e0b; font-size:0.9rem;">
        No features loaded. Please generate or upload churn data and run AutoML first.
      </div>`;
      return;
    }

    dynamicInputs.forEach(el => {
      const featureName = el.dataset.feature;
      payload[featureName] = el.value;
    });

    resDiv.innerHTML = "Analyzing...";
    try {
        const res = await fetch("/api/predict/churn", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const d = await res.json();

        if (!res.ok || d.error) {
            resDiv.innerHTML = `<div style="background:rgba(239,68,68,0.1); border:1px solid #ef4444; padding:15px; border-radius:8px; color:#ef4444; font-size:0.9rem;">
                <b>Analysis Error:</b> ${d.error || "Unknown server error"}
            </div>`;
            return;
        }

        const badge = `<div style="display:inline-block; font-size:0.7rem; background:rgba(96,165,250,0.1); color:#60a5fa; border:1px solid rgba(96,165,250,0.3); padding:2px 6px; border-radius:30px; margin-top:10px;">Model: ${d.model_source}</div>`;
        const resultColor = d.is_positive ? '#10b981' : '#ef4444';
        const probLabel = d.is_churn_inverted ? 'Churn Risk' : 'Confidence';
        resDiv.innerHTML = `<div class="card"><div class="big" style="color:${resultColor}">${d.result_text}</div><div class="muted">${probLabel}: ${d.probability}%</div>${badge}</div>`;
    } catch (err) { 
        resDiv.innerHTML = `<div style="color:#ef4444;">Network error: Could not reach analysis server.</div>`; 
    }
  });

  // -----------------------------------------
  // REPORT DATA
  // -----------------------------------------
  let reportTrendChart = null, reportOrderTrendChart = null, reportDonutChart = null, reportBarChart = null, reportLoyaltyChart = null;
  const reportDataSource = document.getElementById("reportDataSource");
  if (reportDataSource) reportDataSource.addEventListener("change", loadReportData);
  document.getElementById("btnDownloadReport")?.addEventListener("click", () => {
      window.location.href = `/api/report/download?source=${reportDataSource.value}`;
  });

  async function autoSelectBestReport() {
    try {
      const res = await fetch("/api/user_model_status");
      const status = await res.json();
      if (status.has_churn_data && !status.has_sales_data) {
          if (reportDataSource) reportDataSource.value = "churn";
      } else if (status.has_sales_data) {
          if (reportDataSource) reportDataSource.value = "sales";
      }
      loadReportData();
    } catch (e) { loadReportData(); }
  }

  async function loadReportData() {
    const kpiContainer = document.getElementById("kpiContainer");
    const insightsList = document.getElementById("insightsList");
    const source = reportDataSource ? reportDataSource.value : "sales";
    try {
      const res = await fetch(`/api/report/summary?source=${source}`);
      const data = await res.json();
      
      if (data.has_data === false) {
          kpiContainer.innerHTML = `<div style="background:rgba(245,158,11,0.1); border:1px solid #f59e0b; padding:20px; border-radius:12px; grid-column: 1 / -1; text-align:center;">
            <div style="font-size:1.4rem; margin-bottom:8px;">No ${source.toUpperCase()} Data Found</div>
            <div style="color:var(--text-muted); font-size:0.9rem;">To see strategic insights and KPIs, please upload or generate your <b>${source}</b> data in the Home Dashboard first.</div>
          </div>`;
          renderCharts({});
          renderInsights([]);
          return;
      }

      renderKPIs(data.kpis);
      renderCharts(data.charts);
      renderInsights(data.insights);
    } catch (err) { kpiContainer.innerHTML = "Error loading report."; }
  }

  function renderKPIs(kpis) {
      document.getElementById("kpiContainer").innerHTML = Object.entries(kpis).map(([l, v]) => `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value">${v}</div></div>`).join("");
  }

  function renderInsights(ins) {
      const list = document.getElementById("insightsList");
      if (!list) return;
      list.innerHTML = (ins && ins.length) ? ins.map(t => {
          let formatted = t.replace(/\*\*(.*?)\*\*/g, "<b style='color:#60a5fa;'>$1</b>");
          return `<li style="padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05); color:#cbd5e1; font-size:0.95rem; line-height:1.5;">${formatted}</li>`;
      }).join("") : "<li>No insights.</li>";
  }

  function renderCharts(charts) {
      const updateChart = (id, inst, type, cfg) => {
          const el = document.getElementById(id);
          if (!cfg) { el.parentElement.style.display="none"; return inst; }
          el.parentElement.style.display="block";
          if (inst) inst.destroy();
          return new Chart(el.getContext("2d"), { type, data: cfg.data || { labels: cfg.labels, datasets: [{ data: cfg.data }] } , options: cfg.options || { responsive: true } });
      };
      
      // Detailed chart rendering for report
      const cTrend = document.getElementById("cardTrend");
      if (charts.trend) {
          cTrend.style.display = "block";
          reportTrendChart = new Chart(document.getElementById("trendChart").getContext("2d"), {
              type: 'line', 
              data: { 
                  labels: charts.trend.labels, 
                  datasets: [{ 
                      label: charts.trend.label || 'Monthly Revenue', 
                      data: charts.trend.data, 
                      borderColor: '#3b82f6', 
                      tension: 0.3,
                      fill: true,
                      backgroundColor: 'rgba(59, 130, 246, 0.1)'
                  }] 
              },
              options: {
                  responsive: true,
                  plugins: { legend: { display: true } },
                  scales: {
                      y: {
                          ticks: {
                              callback: function(value) { return value.toLocaleString(); }
                          }
                      }
                  }
              }
          });
      } else cTrend.style.display = "none";

      const cOrderTrend = document.getElementById("cardOrderTrend");
      if (charts.order_trend) {
          cOrderTrend.style.display = "block";
          if (reportOrderTrendChart) reportOrderTrendChart.destroy();
          reportOrderTrendChart = new Chart(document.getElementById("orderTrendChart").getContext("2d"), {
              type: 'line', 
              data: { 
                  labels: charts.order_trend.labels, 
                  datasets: [{ 
                      label: charts.order_trend.label || 'Monthly Orders', 
                      data: charts.order_trend.data, 
                      borderColor: '#10b981', 
                      tension: 0.3,
                      fill: true,
                      backgroundColor: 'rgba(16, 185, 129, 0.1)'
                  }] 
              },
              options: {
                  responsive: true,
                  plugins: { legend: { display: true } },
                  scales: {
                      y: {
                          beginAtZero: true,
                          ticks: {
                              callback: function(value) { return value.toLocaleString(); }
                          }
                      }
                  }
              }
          });
      } else if (cOrderTrend) cOrderTrend.style.display = "none";

      const cDonut = document.getElementById("cardDonut");
      if (charts.donut) {
          cDonut.style.display = "block";
          if (reportDonutChart) reportDonutChart.destroy();
          reportDonutChart = new Chart(document.getElementById("categoryChart").getContext("2d"), {
              type: 'doughnut', data: { labels: charts.donut.labels, datasets: [{ data: charts.donut.data, backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }] }
          });
      } else cDonut.style.display = "none";

      const cBar = document.getElementById("cardBar");
      if (charts.bar) {
          cBar.style.display = "block";
          if (reportBarChart) reportBarChart.destroy();
          reportBarChart = new Chart(document.getElementById("cityChart").getContext("2d"), {
              type: 'bar', data: { labels: charts.bar.labels, datasets: [{ label:'Value', data: charts.bar.data, backgroundColor:'#8b5cf6' }] }
          });
      } else cBar.style.display = "none";

      const cLoyalty = document.getElementById("cardLoyalty");
      if (charts.loyalty) {
          cLoyalty.style.display = "block";
          if (reportLoyaltyChart) reportLoyaltyChart.destroy();
          reportLoyaltyChart = new Chart(document.getElementById("loyaltyChart").getContext("2d"), {
              type: 'doughnut', 
              data: { 
                  labels: charts.loyalty.labels, 
                  datasets: [{ 
                      data: charts.loyalty.data, 
                      backgroundColor: ['#94a3b8', '#f59e0b'],
                      borderWidth: 2,
                      borderColor: '#1e293b'
                  }] 
              },
              options: {
                  responsive: true,
                  plugins: { 
                      legend: { position: 'bottom', labels: { color: '#94a3b8' } },
                      title: { display: true, text: charts.loyalty.title, color: '#94a3b8' }
                  },
                  cutout: '70%',
                  maintainAspectRatio: false
              }
          });
      } else if (cLoyalty) {
          cLoyalty.style.display = "none";
      }
  }

});
