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
});
