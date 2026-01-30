document.addEventListener("DOMContentLoaded", () => {
  // -----------------------------
  // Tabs (Demand / Sales)
  // -----------------------------
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // -----------------------------
  // DEMAND: Dependent Sub-Category dropdown
  // -----------------------------
  const dCategory = document.getElementById("d_category");
  const dSubcategory = document.getElementById("d_subcategory");

  async function loadSubcategoriesForCategory(category, targetSelect) {
    try {
      const res = await fetch(
        `/api/options/subcategories?category=${encodeURIComponent(category)}`
      );

      if (!res.ok) {
        throw new Error("Failed to fetch subcategories");
      }

      const data = await res.json();
      const subs = Array.isArray(data.subcategories) ? data.subcategories : [];

      // Replace options
      targetSelect.innerHTML = "";

      if (subs.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No sub-categories found";
        targetSelect.appendChild(opt);
        targetSelect.disabled = true;
        return;
      }

      targetSelect.disabled = false;

      subs.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        targetSelect.appendChild(opt);
      });

    } catch (err) {
      console.error("Failed to load subcategories:", err);
      targetSelect.disabled = false;
    }
  }

  // Demand dropdown behavior
  if (dCategory && dSubcategory) {
    dCategory.addEventListener("change", () => {
      loadSubcategoriesForCategory(dCategory.value, dSubcategory);
    });

    // Initial load on page open
    loadSubcategoriesForCategory(dCategory.value, dSubcategory);
  }

  // -----------------------------
  // DEMAND prediction
  // -----------------------------
  const btnDemand = document.getElementById("btnDemand");
  const demandResult = document.getElementById("demandResult");

  btnDemand?.addEventListener("click", async () => {
    const payload = {
      category: dCategory.value,
      sub_category: dSubcategory.value,
      region: document.getElementById("d_region").value,
      year: document.getElementById("d_year").value,
      month: document.getElementById("d_month").value,
    };

    try {
      const res = await fetch("/api/predict/demand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const out = await res.json();

      if (out.error) {
        demandResult.innerHTML = `<div class="error">${out.error}</div>`;
        return;
      }

      demandResult.innerHTML = `
        <div class="card">
          <div class="big">
            Predicted Monthly Demand (Units): <b>${out.predicted_total_quantity}</b>
          </div>
          <div class="muted">
            Stats source: <b>${out.stats_mode}</b>
          </div>
          <div class="muted">
            Auto-used features:
            Avg Unit Price=${out.used_features.Avg_UnitPrice},
            Avg Discount=${out.used_features.Avg_Discount},
            Orders Count=${out.used_features.Orders_Count}
          </div>
        </div>
      `;
    } catch (err) {
      console.error(err);
      demandResult.innerHTML = `<div class="error">Server error while predicting demand.</div>`;
    }
  });

  // -----------------------------
  // SALES: dependent subcategory too (uses same API)
  // -----------------------------
  const sCategory = document.getElementById("s_category");
  const sSubcategory = document.getElementById("s_subcategory");

  if (sCategory && sSubcategory) {
    sCategory.addEventListener("change", () => {
      loadSubcategoriesForCategory(sCategory.value, sSubcategory);
    });

    // Initial load for sales
    loadSubcategoriesForCategory(sCategory.value, sSubcategory);
  }

  // -----------------------------
  // SALES prediction (NO year/month/quarter)
  // -----------------------------
  const btnSales = document.getElementById("btnSales");
  const salesResult = document.getElementById("salesResult");

  // ✅ Safety check required by you
  if (!btnSales) {
    console.error("Sales button not found");
    return;
  }

  btnSales.addEventListener("click", async () => {
    const payload = {
      category: document.getElementById("s_category").value,
      sub_category: document.getElementById("s_subcategory").value,
      region: document.getElementById("s_region").value,
      city: document.getElementById("s_city").value,
      unit_price: document.getElementById("s_unitprice").value,
      discount: document.getElementById("s_discount").value,
      quantity: document.getElementById("s_quantity").value,
    };

    // ✅ Error handling required by you
    try {
      const res = await fetch("/api/predict/sales", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const out = await res.json();

      if (!res.ok || out.error) {
        salesResult.innerHTML = `<div class="error">${out.error || "Sales prediction failed."}</div>`;
        return;
      }

      salesResult.innerHTML = `
        <div class="card">
          <div class="big">
            Predicted Sales (Revenue): <b>${out.predicted_sales}</b>
          </div>
          <div class="muted">
            Per-order revenue estimate for the given inputs.
          </div>
        </div>
      `;
    } catch (err) {
      console.error(err);
      salesResult.innerHTML = `<div class="error">Error: ${err.message}</div>`;
    }
  });


  // -----------------------------
  // FORECASTING Logic
  // -----------------------------
  const btnForecast = document.getElementById("btnForecast");
  const forecastResult = document.getElementById("forecastResult");
  const ctx = document.getElementById("forecastChart")?.getContext("2d");
  let chartInstance = null;

  if (btnForecast) {
    btnForecast.addEventListener("click", async () => {
      const category = document.getElementById("f_category").value;
      const horizon = document.getElementById("f_horizon").value;

      try {
        // Show loading state
        if (forecastResult) {
          const metricsDiv = document.getElementById("forecastMetrics");
          if (metricsDiv) metricsDiv.innerHTML = "Loading...";
        }

        const res = await fetch("/api/forecast/sales_series", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ category, horizon })
        });

        const data = await res.json();

        if (data.error) {
          if (forecastResult) forecastResult.innerHTML = `<div class="error">${data.error}</div>`;
          return;
        }

        // Render Chart
        if (chartInstance) {
          chartInstance.destroy();
        }

        const historyDates = data.history.dates;
        const historyValues = data.history.values;
        const forecastDates = data.forecast.dates;
        const forecastValues = data.forecast.values;

        // Combine for plotting
        // We need nulls for the "future" part of the history line, and nulls for the "past" part of the forecast line
        // to make them look distinct.

        // Simpler approach: just two datasets on the same x-axis labels
        const allLabels = [...historyDates, ...forecastDates];

        // Pad history with nulls for future
        const plotHistory = [...historyValues, ...new Array(forecastDates.length).fill(null)];

        // Pad forecast with nulls for past (except connect to last history point?)
        // To connect lines, the first point of forecast should ideally match last point of history.
        // For simplicity now, let's just pad.
        const plotForecast = [...new Array(historyDates.length - 1).fill(null), historyValues[historyValues.length - 1], ...forecastValues];

        chartInstance = new Chart(ctx, {
          type: 'line',
          data: {
            labels: allLabels,
            datasets: [
              {
                label: 'Historical Sales',
                data: plotHistory,
                borderColor: '#3b82f6', // Blue
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3
              },
              {
                label: 'Forecast',
                data: plotForecast,
                borderColor: '#10b981', // Green
                borderDash: [5, 5],
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.3
              }
            ]
          },
          options: {
            responsive: true,
            interaction: {
              intersect: false,
              mode: 'index',
            },
            plugins: {
              title: {
                display: true,
                text: `Sales Forecast (${category})`
              },
              legend: {
                labels: { color: '#cbd5e1' }
              }
            },
            scales: {
              x: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#334155' }
              },
              y: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#334155' }
              }
            }
          }
        });

        // Update Metrics
        const metricsDiv = document.getElementById("forecastMetrics");
        if (metricsDiv) {
          metricsDiv.innerHTML = `
            <div><b>Best Model Selected:</b> ${data.model_name}</div>
            <div><b>Best Model Selected:</b> ${data.model_name}</div>
            <div><b>Model Error (MAPE):</b> ${data.mape}%</div>
          `;
        }

      } catch (err) {
        console.error(err);
        if (forecastResult) forecastResult.innerHTML = `<div class="error">Error generating forecast.</div>`;
      }
    });
  }

});
