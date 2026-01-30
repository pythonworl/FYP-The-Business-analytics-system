console.log("App.js script started");

document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM fully loaded");

  // -----------------------------
  // Tabs (Demand / Sales)
  // -----------------------------
  const tabs = document.querySelectorAll(".tab");
  if (tabs.length === 0) console.warn("No tabs found");

  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(btn.dataset.tab);
      if (targetPanel) {
        targetPanel.classList.add("active");
      } else {
        console.error(`Panel with id '${btn.dataset.tab}' not found`);
      }
    });
  });

  // -----------------------------
  // Shared: fetch subcategories
  // -----------------------------
  async function fetchSubcategories(category) {
    const res = await fetch(`/api/options/subcategories?category=${encodeURIComponent(category)}`);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Subcategory API failed (${res.status}). ${text}`);
    }
    const data = await res.json();
    return Array.isArray(data.subcategories) ? data.subcategories : [];
  }

  function populateSelect(selectEl, options, emptyText = "No sub-categories found") {
    selectEl.innerHTML = "";

    if (!options || options.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = emptyText;
      selectEl.appendChild(opt);
      selectEl.disabled = true;
      return;
    }

    selectEl.disabled = false;
    options.forEach((val) => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = val;
      selectEl.appendChild(opt);
    });
  }

  // -----------------------------
  // DEMAND: Dependent Sub-Category dropdown
  // -----------------------------
  const dCategory = document.getElementById("d_category");
  const dSubcategory = document.getElementById("d_subcategory");

  async function loadDemandSubcategories() {
    if (!dCategory || !dSubcategory) return;
    try {
      const subs = await fetchSubcategories(dCategory.value);
      populateSelect(dSubcategory, subs);
    } catch (err) {
      console.error("Demand subcategory load failed:", err);
      dSubcategory.disabled = false; // keep usable
    }
  }

  if (dCategory && dSubcategory) {
    dCategory.addEventListener("change", loadDemandSubcategories);
    // Initial load
    loadDemandSubcategories();
  } else {
    console.warn("Demand category elements missing");
  }

  // -----------------------------
  // SALES: Dependent Sub-Category dropdown
  // -----------------------------
  const sCategory = document.getElementById("s_category");
  const sSubcategory = document.getElementById("s_subcategory");

  async function loadSalesSubcategories() {
    if (!sCategory || !sSubcategory) return;
    try {
      const subs = await fetchSubcategories(sCategory.value);
      populateSelect(sSubcategory, subs);
    } catch (err) {
      console.error("Sales subcategory load failed:", err);
      sSubcategory.disabled = false;
    }
  }

  if (sCategory && sSubcategory) {
    sCategory.addEventListener("change", loadSalesSubcategories);
    // Initial load
    loadSalesSubcategories();
  } else {
    console.warn("Sales category elements missing");
  }

  // -----------------------------
  // DEMAND prediction
  // -----------------------------
  const btnDemand = document.getElementById("btnDemand");
  const demandResult = document.getElementById("demandResult");

  if (btnDemand) {
    btnDemand.addEventListener("click", async () => {
      const regionEl = document.getElementById("d_region");
      const yearEl = document.getElementById("d_year");
      const monthEl = document.getElementById("d_month");

      const payload = {
        category: dCategory?.value || "",
        sub_category: dSubcategory?.value || "",
        region: regionEl?.value || "",
        year: yearEl?.value || "",
        month: monthEl?.value || "",
      };

      try {
        const res = await fetch("/api/predict/demand", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const out = await res.json().catch(() => ({}));

        if (!res.ok) {
          if (demandResult) demandResult.innerHTML = `<div class="error">Demand API error (${res.status}): ${out.error || "Unknown error"}</div>`;
          return;
        }

        if (out.error) {
          if (demandResult) demandResult.innerHTML = `<div class="error">${out.error}</div>`;
          return;
        }

        if (demandResult) {
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
        }
      } catch (err) {
        console.error(err);
        if (demandResult) demandResult.innerHTML = `<div class="error">Server error while predicting demand.</div>`;
      }
    });
  } else {
    console.warn("btnDemand not found");
  }


  // -----------------------------
  // SALES prediction
  // -----------------------------
  const btnSales = document.getElementById("btnSales");
  const salesResult = document.getElementById("salesResult");

  if (btnSales) {
    console.log("Sales button found, attaching listener");
    btnSales.addEventListener("click", async () => {
      console.log("Sales prediction button clicked");

      const sCategory = document.getElementById("s_category");
      const sSubcategory = document.getElementById("s_subcategory");
      const sRegion = document.getElementById("s_region");
      const sCity = document.getElementById("s_city");
      const sUnitPrice = document.getElementById("s_unitprice");
      const sDiscount = document.getElementById("s_discount");
      const sQuantity = document.getElementById("s_quantity");

      // Validate existence
      if (!sCategory || !sSubcategory || !sRegion || !sCity || !sUnitPrice || !sDiscount || !sQuantity) {
        console.error("One or more input elements are missing");
        if (salesResult) salesResult.innerHTML = `<div class="error">Error: Missing input fields. Please refresh page.</div>`;
        return;
      }

      const payload = {
        category: sCategory.value,
        sub_category: sSubcategory.value,
        region: sRegion.value,
        city: sCity.value,
        unit_price: sUnitPrice.value,
        discount: sDiscount.value,
        quantity: sQuantity.value,
      };

      console.log("Sending payload:", payload);

      try {
        if (salesResult) salesResult.innerHTML = `<div class="muted">Predicting...</div>`;

        const res = await fetch("/api/predict/sales", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        console.log("Response status:", res.status);

        if (!res.ok) {
          const out = await res.json().catch(() => ({}));
          throw new Error(`Server returned status ${res.status}: ${out.error || ''}`);
        }

        const out = await res.json();
        console.log("Response data:", out);

        if (out.error) {
          if (salesResult) salesResult.innerHTML = `<div class="error">${out.error}</div>`;
          return;
        }

        if (salesResult) {
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
        }
      } catch (err) {
        console.error("Sales prediction error:", err);
        if (salesResult) salesResult.innerHTML = `<div class="error">Error: ${err.message}</div>`;
      }
    });
  } else {
    // This is the critical log for the user's issue
    console.error("btnSales element not found! Event listener NOT attached.");
    if (salesResult) salesResult.innerHTML = `<div class="error">Error: Predict button not found (Internal Error)</div>`;
  }

});
