document.addEventListener("DOMContentLoaded", () => {
  console.log("Business Analytics App: Initializing...");

  // UTILS
  function safeInit(name, fn) {
    try {
      fn();
      console.log(`Module loaded: ${name}`);
    } catch (e) {
      console.error(`Module FAILED: ${name}`, e);
    }
  }

  // 1. TABS
  safeInit("Tabs", () => {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const panel = document.getElementById(btn.dataset.tab);
        if (panel) panel.classList.add("active");
        if (btn.dataset.tab === "report") loadReportData();
      });
    });
  });

  // 2. INTELLIGENCE CHAT (Moved up for priority)
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
      if (data.error) appendMessage("⚠️ Error: " + data.error, "bot");
      else appendMessage(data.reply, "bot");
    } catch (err) {
      console.error(err);
      document.getElementById(loadingId)?.remove();
      appendMessage("❌ Connection error. Please try again.", "bot");
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
      appendMessage("System ready. How can I help?", "bot");
    }
  });

  // 3. DROPDOWNS & PREDICTIONS (Wrapped in safeInit)
  const dCategory = document.getElementById("d_category");
  const dSubcategory = document.getElementById("d_subcategory");
  const sCategory = document.getElementById("s_category");
  const sSubcategory = document.getElementById("s_subcategory");

  async function loadSubcategoriesForCategory(category, targetSelect) {
    if (!targetSelect) return;
    try {
      const res = await fetch(`/api/options/subcategories?category=${encodeURIComponent(category)}`);
      const data = await res.json();
      const subs = data.subcategories || [];
      targetSelect.innerHTML = "";
      if (subs.length === 0) {
        targetSelect.innerHTML = '<option value="">No sub-categories found</option>';
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
    document.getElementById("btnDemand")?.addEventListener("click", async () => {
      const demandResult = document.getElementById("demandResult");
      const payload = {
        category: dCategory.value, sub_category: dSubcategory.value,
        region: document.getElementById("d_region").value,
        year: document.getElementById("d_year").value,
        month: document.getElementById("d_month").value,
      };
      try {
        const res = await fetch("/api/predict/demand", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const out = await res.json();
        if (out.error) { demandResult.innerHTML = `<div class="error">${out.error}</div>`; return; }
        demandResult.innerHTML = `<div class="card"><div class="big">Predicted Monthly Demand (Units): <b>${out.predicted_total_quantity}</b></div><div class="muted">Stats source: <b>${out.stats_mode}</b></div></div>`;
      } catch (err) { demandResult.innerHTML = '<div class="error">Predict error</div>'; }
    });
  });

  safeInit("Sales", () => {
    if (sCategory && sSubcategory) {
      sCategory.addEventListener("change", () => loadSubcategoriesForCategory(sCategory.value, sSubcategory));
      loadSubcategoriesForCategory(sCategory.value, sSubcategory);
    }
    document.getElementById("btnSales")?.addEventListener("click", async () => {
      const salesResult = document.getElementById("salesResult");
      const payload = {
        category: sCategory.value, sub_category: sSubcategory.value,
        region: document.getElementById("s_region").value, city: document.getElementById("s_city").value,
        unit_price: document.getElementById("s_unitprice").value,
        discount: document.getElementById("s_discount").value, quantity: document.getElementById("s_quantity").value,
      };
      try {
        const res = await fetch("/api/predict/sales", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const out = await res.json();
        salesResult.innerHTML = `<div class="card"><div class="big">Predicted Sales: <b>${out.predicted_sales}</b></div></div>`;
      } catch (err) { salesResult.innerHTML = '<div class="error">Predict error</div>'; }
    });
  });

  // 4. FORECASTING & REPORT (Condensed and safe)
  safeInit("Forecast", () => {
    let chartInstance = null;
    document.getElementById("btnForecast")?.addEventListener("click", async () => {
      const ctx = document.getElementById("forecastChart")?.getContext("2d");
      const res = await fetch("/api/forecast/sales_series", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category: document.getElementById("f_category").value, horizon: document.getElementById("f_horizon").value }) });
      const data = await res.json();
      if (chartInstance) chartInstance.destroy();
      const allLabels = [...data.history.dates, ...data.forecast.dates];
      const plotHistory = [...data.history.values, ...new Array(data.forecast.dates.length).fill(null)];
      const plotForecast = [...new Array(data.history.dates.length - 1).fill(null), data.history.values[data.history.values.length - 1], ...data.forecast.values];
      chartInstance = new Chart(ctx, { type: 'line', data: { labels: allLabels, datasets: [{ label: 'Historical', data: plotHistory, borderColor: '#3b82f6' }, { label: 'Forecast', data: plotForecast, borderColor: '#10b981', borderDash: [5, 5] }] } });
      document.getElementById("forecastMetrics").innerHTML = `<div>Model: ${data.model_name} (MAPE: ${data.mape}%)</div>`;
    });
  });

  // CHURN
  safeInit("Churn", () => {
    document.getElementById("btnChurn")?.addEventListener("click", async () => {
      const churnResult = document.getElementById("churnResult");
      const payload = {
        Age: document.getElementById("c_age").value, AnnualIncome: document.getElementById("c_income").value,
        NumberOfPurchases: document.getElementById("c_purchases").value, TimeSpentOnWebsite: document.getElementById("c_time").value,
        CustomerTenureYears: document.getElementById("c_tenure").value, LastPurchaseDaysAgo: document.getElementById("c_last_purchase").value,
        SessionCount: document.getElementById("c_sessions").value, CustomerSatisfaction: document.getElementById("c_satisfaction").value,
        DiscountsAvailed: document.getElementById("c_discounts").value, LoyaltyProgram: document.getElementById("c_loyalty").value
      };
      const res = await fetch("/api/predict/churn", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const d = await res.json();
      churnResult.innerHTML = `<div class="card"><div class="big" style="color:${d.prediction === 1 ? '#10b981' : '#ef4444'}">${d.result_text}</div><div class="muted">Prob: ${d.probability}%</div></div>`;
    });
  });

  // REPORT
  let trendChartInstance = null;
  let categoryChartInstance = null;
  let cityChartInstance = null;

  async function loadReportData() {
    const kpiContainer = document.getElementById("kpiContainer");
    if (!kpiContainer) return;

    try {
      const res = await fetch("/api/report/summary");
      const data = await res.json();

      renderKPIs(data.kpis);
      renderTrendChart(data.trend);
      renderSizeChart(data.size_share);
      renderColorChart(data.color_share);
      renderInsights(data.insights);

    } catch (err) {
      console.error("Report Load Error:", err);
      kpiContainer.innerHTML = '<div class="error">Error loading report data.</div>';
    }
  }

  function renderKPIs(kpis) {
    const kpiContainer = document.getElementById("kpiContainer");
    kpiContainer.innerHTML = `
            <div class="kpi-card"><div class="kpi-label">Total Sales</div><div class="kpi-value">${kpis.total_sales.toLocaleString()}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Orders</div><div class="kpi-value">${kpis.total_orders.toLocaleString()}</div></div>
            <div class="kpi-card"><div class="kpi-label">Qty Sold</div><div class="kpi-value">${kpis.total_quantity.toLocaleString()}</div></div>
            <div class="kpi-card"><div class="kpi-label">Avg Order Value</div><div class="kpi-value">${kpis.aov}</div></div>
        `;
  }

  function renderInsights(insights) {
    const list = document.getElementById("insightsList");
    if (!list) return;
    list.innerHTML = (insights && insights.length)
      ? insights.map(text => `<li>${text}</li>`).join("")
      : "<li>No specific insights available.</li>";
  }

  function renderTrendChart(trendData) {
    const canvas = document.getElementById("trendChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (trendChartInstance) trendChartInstance.destroy();
    trendChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: trendData.labels,
        datasets: [{ label: 'Sales', data: trendData.sales, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, tension: 0.3 }]
      },
      options: { responsive: true, plugins: { legend: { labels: { color: '#cbd5e1' } } } }
    });
  }

  function renderSizeChart(sizeData) {
    const canvas = document.getElementById("categoryChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (categoryChartInstance) categoryChartInstance.destroy();
    categoryChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: sizeData.labels,
        datasets: [{ data: sizeData.data, backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'], borderWidth: 0 }]
      },
      options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#cbd5e1' } } } }
    });
  }

  function renderColorChart(colorData) {
    const canvas = document.getElementById("cityChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (cityChartInstance) cityChartInstance.destroy();
    cityChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: colorData.labels,
        datasets: [{ label: 'Sales', data: colorData.data, backgroundColor: '#8b5cf6', borderRadius: 4 }]
      },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });
  }
});
