// Test-split confusion matrices of the last training run: model -> {labels, matrix}
var confusionMatrixData = {};

// Escape text before it is inserted into HTML (S5). CSV headers, cell values,
// class names, error messages and predictions all come from user-controlled data.
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Number for display: fixed decimals, or "n/a" when missing
function fmt(value, digits) {
  return value === null || value === undefined || isNaN(value) ? "n/a" : Number(value).toFixed(digits);
}

// Checkbox lists replace Ctrl+Click multi-selects, so they work on touch screens (U5).
function fillCheckboxList(container, items) {
  container.innerHTML = "";
  items.forEach(item => {
    const label = document.createElement("label");
    label.className = "checkbox-item";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = item.value;
    box.checked = Boolean(item.checked);
    const text = document.createElement("span");
    text.textContent = item.text;
    if (item.title) {
      text.title = item.title;  // tooltip only: the checkbox keeps its visible text as its name
    }
    label.appendChild(box);
    label.appendChild(text);
    container.appendChild(label);
  });
}

function checkedValues(container) {
  return Array.from(container.querySelectorAll("input[type=checkbox]:checked")).map(box => box.value);
}

function setChecked(container, values) {
  container.querySelectorAll("input[type=checkbox]").forEach(box => {
    box.checked = values.includes(box.value);
  });
}

const BALANCING_NAMES = {
  "none": "None (No Balancing)",
  "recommended": "Recommended",
  "class_weight": "Class Weights",
  "smote": "SMOTE",
  "smote_enn": "SMOTE-ENN",
  "adasyn": "ADASYN",
  "borderline_smote": "Borderline-SMOTE",
  "random_undersample": "Random Undersampling",
  "tomek": "Tomek Links",
  "smote_tomek": "SMOTE + Tomek Links"
};

// Plain-language text for the split's reason and warning codes (toolkit/ids_core/split.py)
const SPLIT_MESSAGES = {
  "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC": "Rows were split at random (stratified by class), so flows of the same attack can be on both sides: the scores may be optimistic.",
  "CLASS_MISSING_FROM_TEST": "Some classes have no rows in the test split, so they are not tested.",
  "CLASS_MISSING_FROM_TRAIN": "Some classes appear only in the test split; no model can learn them.",
  "OFFICIAL_TEST_OVERLAP_KEPT": "Some rows of the official test file also occur in the training file. They are kept as published, so results stay comparable with other work.",
  "OFFICIAL_COPIES_KEPT": "The official files contain repeated rows; they are kept as published.",
  "OFFICIAL_TEST_OVERLAP_REMOVED": "Test rows that also occur in the training file were removed (strict mode).",
  "TIME_MISSING_ROWS_DROPPED": "Rows without a time value were left out of the time-ordered split.",
  "LABEL_MISSING_ROWS_DROPPED": "Rows without a label were left out.",
  "OFFICIAL_TEST_FILE_MISSING": "No official test file was uploaded.",
  "TIME_COLUMN_MISSING": "There is no usable time column.",
  "TIME_MISSING_RATIO_EXCEEDED": "Too many rows have no time value.",
  "TIME_PARSE_FAILURE_RATIO_EXCEEDED": "Too many time values could not be read.",
  "TIME_FORMAT_AMBIGUOUS": "The date format is ambiguous (day/month order).",
  "TIME_NON_MONOTONIC_ACROSS_FILES": "The files' time ranges overlap or go backwards.",
  "TIME_TOO_COARSE_FOR_CLASS": "A class happens within too short a time to split by time.",
  "GROUP_COLUMN_MISSING": "There is no group column.",
  "GROUP_TOO_FEW_GROUPS": "There are too few groups to split by group.",
  "GROUP_CLASS_ONE_SIDED": "A class occurs in one group only."
};

function splitMessage(code) {
  return SPLIT_MESSAGES[code] || code;
}

const BALANCING_HINTS = {
  "none": "No balancing will be applied. Use this if your dataset is already balanced.",
  "recommended": "The toolkit picks a method from the <strong>training split's</strong> class counts when you train.",
  "class_weight": "Fast and memory-efficient. Adjusts model weights to give more importance to minority classes.",
  "smote": "Creates synthetic samples for minority classes. Good for continuous features. May increase training time.",
  "smote_enn": "SMOTE + Edited Nearest Neighbors. Creates synthetic samples then removes noisy ones.",
  "adasyn": "Adaptive SMOTE - focuses on harder-to-learn samples.",
  "borderline_smote": "Only generates synthetic samples near class boundaries. Good when classes overlap.",
  "random_undersample": "Removes random samples from bigger classes. Very fast but may lose important data.",
  "tomek": "Removes overlapping samples between classes. Cleans decision boundaries without adding data.",
  "smote_tomek": "SMOTE + Tomek Links. Creates synthetic samples then cleans boundaries."
};


document.addEventListener("DOMContentLoaded", function() {

  /********** DYNAMIC IDS FUNCTIONALITY **********/
  if (document.getElementById("uploadForm")) {
    // Model titles from /get_available_models (id -> title)
    var modelTitles = {};
    function modelTitle(id) {
      return modelTitles[id] || id;
    }

    // Test split size from the slider (0.1 - 0.5)
    function testSize() {
      return parseInt(document.getElementById("trainTestSlider").value) / 100;
    }

    // Reset file inputs
    function resetFileInputs(formId) {
      const form = document.getElementById(formId);
      if (form) {
        form.reset();
        const fileInputs = form.querySelectorAll('input[type="file"]');
        fileInputs.forEach(function(input) {
          input.value = "";
          const placeholder = input.parentNode.querySelector('.file-input-placeholder');
          if (placeholder) {
            placeholder.textContent = input.multiple ? "Choose file(s)..." : "Choose file...";
          }
        });
      }
    }

    // Data Preview Function
    function fetchDataPreview() {
      const numRows = document.getElementById("previewRows").value || 10;
      const dataStats = document.getElementById("dataStats");
      const dataPreviewTable = document.getElementById("dataPreviewTable");

      dataStats.innerHTML = "Loading preview...";
      dataPreviewTable.innerHTML = "";

      fetch("/preview_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_rows: parseInt(numRows) })
      })
      .then(response => response.json())
      .then(result => {
        if (result.preview) {
          // Display statistics
          const stats = result.stats;
          dataStats.innerHTML = `
            <div class="stats-grid">
              <div class="stat-item"><strong>Total Rows:</strong> ${escapeHtml(stats.total_rows.toLocaleString())}</div>
              <div class="stat-item"><strong>Total Columns:</strong> ${escapeHtml(stats.total_columns)}</div>
              <div class="stat-item"><strong>Numeric Columns:</strong> ${escapeHtml(stats.numeric_columns)}</div>
              <div class="stat-item"><strong>Categorical Columns:</strong> ${escapeHtml(stats.categorical_columns)}</div>
            </div>
          `;

          // Build preview table
          let tableHTML = '<table class="preview-table"><thead><tr>';
          result.columns.forEach(col => {
            tableHTML += `<th>${escapeHtml(col)}</th>`;
          });
          tableHTML += '</tr></thead><tbody>';

          result.preview.forEach(row => {
            tableHTML += '<tr>';
            result.columns.forEach(col => {
              let cellValue = row[col];
              // Truncate long values
              if (typeof cellValue === 'string' && cellValue.length > 30) {
                cellValue = cellValue.substring(0, 27) + '...';
              }
              tableHTML += `<td>${escapeHtml(cellValue !== null && cellValue !== undefined ? cellValue : '')}</td>`;
            });
            tableHTML += '</tr>';
          });

          tableHTML += '</tbody></table>';
          dataPreviewTable.innerHTML = tableHTML;
        } else {
          dataStats.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        dataStats.textContent = "Error loading preview: " + error;
      });
    }

    // Refresh preview button handler
    document.getElementById("refreshPreviewBtn").addEventListener("click", fetchDataPreview);

    // Fill the target dropdowns and the feature list; the profile's label is preselected
    function populateColumns(columns, dataset) {
      const profile = dataset.profile;
      const label = profile.label_column;
      const blocked = profile.blocked_columns || [];
      ["targetColumn", "targetForRec", "targetForDist"].forEach(id => {
        const select = document.getElementById(id);
        select.innerHTML = "";
        columns.forEach(col => {
          const option = document.createElement("option");
          option.value = col;
          option.text = col;
          option.selected = col === label;
          select.appendChild(option);
        });
      });
      fillCheckboxList(document.getElementById("inputFeatures"), columns
        .filter(col => col !== label)
        .map(col => ({ value: col, text: col, checked: !blocked.includes(col),
                       title: blocked.includes(col) ? "Blocked by the dataset profile" : "" })));
    }

    // The target can't also be an input feature
    document.getElementById("targetColumn").addEventListener("change", function() {
      const list = document.getElementById("inputFeatures");
      const target = this.value;
      list.querySelectorAll("input[type=checkbox]").forEach(box => {
        if (box.value === target) {
          box.checked = false;
        }
      });
    });

    // "Select all" / "Clear" buttons of a checkbox list
    document.querySelectorAll("button[data-list]").forEach(button => {
      button.addEventListener("click", function() {
        const list = document.getElementById(this.dataset.list);
        const target = document.getElementById("targetColumn").value;
        list.querySelectorAll("input[type=checkbox]").forEach(box => {
          box.checked = this.dataset.check === "all" && box.value !== target;
        });
      });
    });

    // Function to load available models for selection
    function loadAvailableModels() {
      const list = document.getElementById("modelSelection");
      list.innerHTML = "";
      fetch("/get_available_models")
        .then(response => response.json())
        .then(result => {
          if (result.models) {
            result.models.forEach(model => {
              modelTitles[model.id] = model.name;
            });
            fillCheckboxList(list, result.models.map(model => ({
              value: model.id, text: model.name, title: model.description, checked: model.default })));
          }
        })
        .catch(error => {
          console.error("Error loading models:", error);
        });
    }

    // Store current target column for filtering
    var currentTargetColumn = null;

    // Classes that can be removed, smallest first
    function populateTargetValueFilter(classCounts) {
      const sortedClasses = Object.entries(classCounts).sort((a, b) => a[1] - b[1]);
      fillCheckboxList(document.getElementById("targetValuesToRemove"), sortedClasses.map(
        ([className, count]) => ({ value: className, text: className + " (" + count.toLocaleString() + " samples)" })));
    }

    // Remove target values handler
    document.getElementById("removeTargetValuesBtn").addEventListener("click", function() {
      const selectedValues = checkedValues(document.getElementById("targetValuesToRemove"));
      const resultDiv = document.getElementById("targetFilterResult");

      if (selectedValues.length === 0) {
        resultDiv.innerHTML = "<span style='color: #ffc107;'>⚠️ Please tick at least one value to remove, or click 'Skip' to keep all values.</span>";
        return;
      }

      if (!currentTargetColumn) {
        resultDiv.innerHTML = "<span style='color: #dc3545;'>❌ Error: Target column not set. Please analyze class distribution first.</span>";
        return;
      }

      resultDiv.innerHTML = "Removing selected values...";

      fetch("/filter_target_values", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: currentTargetColumn,
          values_to_remove: selectedValues
        })
      })
      .then(response => response.json())
      .then(result => {
        if (result.success) {
          resultDiv.innerHTML = `<span style='color: #28a745;'>✅ ${escapeHtml(result.message)}</span>`;
          populateTargetValueFilter(result.new_class_counts);
          fetchDataPreview();
          document.getElementById("analyzeClassBtn").click();
        } else {
          resultDiv.innerHTML = `<span style='color: #dc3545;'>❌ Error: ${escapeHtml(result.error)}</span>`;
        }
      })
      .catch(error => {
        resultDiv.innerHTML = `<span style='color: #dc3545;'>❌ Error: ${escapeHtml(error)}</span>`;
      });
    });

    // Skip target filter handler
    document.getElementById("skipTargetFilterBtn").addEventListener("click", function() {
      const resultDiv = document.getElementById("targetFilterResult");
      resultDiv.innerHTML = "<span style='color: #17a2b8;'>ℹ️ Keeping all target values.</span>";
      document.getElementById("smoteOption").style.display = "block";
    });

    // Train/Test Split Slider Handler
    const trainTestSlider = document.getElementById("trainTestSlider");
    const splitLabel = document.getElementById("splitLabel");
    trainTestSlider.addEventListener("input", function() {
      splitLabel.textContent = `Train: ${100 - parseInt(this.value)}% | Test: ${parseInt(this.value)}%`;
    });

    // Balancing Method Hint Handler
    const balancingMethodSelect = document.getElementById("balancingMethod");
    const balancingHint = document.getElementById("balancingHint");
    balancingMethodSelect.addEventListener("change", function() {
      balancingHint.innerHTML = BALANCING_HINTS[this.value] || "Select a method to handle class imbalance.";
    });

    // Balancing preview on the training split (/apply_smote changes nothing)
    document.getElementById("previewBalancingBtn").addEventListener("click", function() {
      const resultDiv = document.getElementById("balancingPreviewResult");
      const target = currentTargetColumn || document.getElementById("targetForDist").value;
      const target_features = checkedValues(document.getElementById("inputFeatures")).filter(f => f !== target);
      resultDiv.innerHTML = "Running the balancing method on the training split...";
      fetch("/apply_smote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: target, balancing_method: balancingMethodSelect.value,
                               test_size: testSize(), input_features: target_features })
      })
      .then(response => response.json())
      .then(result => {
        if (result.before) {
          let html = `<div class='info-message'>${escapeHtml(result.message)}</div>`;
          html += `<p><strong>${escapeHtml(BALANCING_NAMES[result.balancer] || result.balancer)}</strong> on ${escapeHtml(result.rows.toLocaleString())} training rows`;
          if (result.sampled) {
            html += ` (a stratified sample of ${escapeHtml(result.train_rows.toLocaleString())})`;
          }
          html += ":</p><table class='metrics-table'><thead><tr><th>Class</th><th>Before</th><th>After</th></tr></thead><tbody>";
          Object.keys(result.before).forEach(cls => {
            html += `<tr><td>${escapeHtml(cls)}</td><td>${escapeHtml(result.before[cls].toLocaleString())}</td><td>${escapeHtml((result.after[cls] || 0).toLocaleString())}</td></tr>`;
          });
          html += "</tbody></table>";
          if (!result.changes_rows) {
            html += "<p>The number of rows does not change with this method.</p>";
          }
          resultDiv.innerHTML = html;
        } else {
          resultDiv.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        resultDiv.textContent = "Error: " + error;
      });
    });

    // Confusion Matrix Display Function
    function displayConfusionMatrix(matrix, labels, modelName) {
      const container = document.getElementById("confusionMatrixDisplay");

      if (!matrix || matrix.length === 0) {
        container.innerHTML = "<p>No confusion matrix data available.</p>";
        return;
      }

      let html = `<div class="cm-header"><strong>${escapeHtml(modelTitle(modelName))}</strong> - Test Set</div>`;
      html += '<table class="confusion-matrix">';

      // Header row with predicted labels
      html += '<tr><th></th><th colspan="' + escapeHtml(matrix.length) + '">Predicted</th></tr>';
      html += '<tr><th></th>';

      // Use labels if available, otherwise use indices
      const displayLabels = labels && labels.length === matrix.length ? labels :
                           matrix.map((_, i) => `Class ${escapeHtml(i)}`);

      displayLabels.forEach(label => {
        // Truncate long labels
        const truncLabel = typeof label === 'string' && label.length > 12 ?
                          label.substring(0, 10) + '...' : label;
        html += `<th title="${escapeHtml(label)}">${escapeHtml(truncLabel)}</th>`;
      });
      html += '</tr>';

      // Data rows
      matrix.forEach((row, i) => {
        html += '<tr>';
        const rowLabel = displayLabels[i];
        const truncLabel = typeof rowLabel === 'string' && rowLabel.length > 12 ?
                          rowLabel.substring(0, 10) + '...' : rowLabel;
        html += `<th title="${escapeHtml(rowLabel)}">${escapeHtml(truncLabel)}</th>`;

        // Calculate row total for percentage
        const rowTotal = row.reduce((sum, val) => sum + val, 0);

        row.forEach((val, j) => {
          // Highlight diagonal (correct predictions)
          const isCorrect = i === j;
          const share = rowTotal > 0 ? val / rowTotal : 0;
          const percentage = (share * 100).toFixed(1);
          const bgColor = isCorrect ?
            `rgba(0, 128, 0, ${escapeHtml(Math.min(share, 0.5))})` :  // max 0.5: dark text stays >= 4.5:1
            `rgba(255, 0, 0, ${escapeHtml(Math.min(share, 0.5))})`;
          html += `<td style="background-color: ${escapeHtml(bgColor)}; color: black" title="${escapeHtml(val)} (${escapeHtml(percentage)}%)">${escapeHtml(val)}</td>`;
        });
        html += '</tr>';
      });

      html += '</table>';

      // Add legend
      html += '<div class="cm-legend"><span class="cm-correct">&#9632; Correct</span> <span class="cm-incorrect">&#9632; Incorrect</span></div>';

      container.innerHTML = html;
    }

    // Show Confusion Matrix Button Handler
    document.getElementById("showConfusionMatrixBtn").addEventListener("click", function() {
      const modelName = document.getElementById("cmModelSelect").value;
      if (!modelName) {
        document.getElementById("confusionMatrixDisplay").innerHTML = "<p>Please select a model.</p>";
        return;
      }
      const data = confusionMatrixData[modelName];
      if (data) {
        displayConfusionMatrix(data.matrix, data.labels, modelName);
      } else {
        document.getElementById("confusionMatrixDisplay").innerHTML = "<p>No confusion matrix for this model.</p>";
      }
    });

    // Toggle sections based on radio selection
    function toggleSections() {
      const trainedOption = document.getElementById("optionTrained");
      const notTrainedOption = document.getElementById("optionNotTrained");
      const loadModelSection = document.getElementById("loadModelSection");
      const uploadDatasetSection = document.getElementById("uploadDatasetSection");
      const recommendationSection = document.getElementById("recommendationSection");
      const configSection = document.getElementById("configSection");
      const predictionSection = document.getElementById("predictionSection");
      const exportSection = document.getElementById("exportSection");
      const loadedFeaturesSection = document.getElementById("loadedFeaturesSection");
      const classImbalanceSection = document.getElementById("classImbalanceSection");

      if (trainedOption.checked) {
        resetFileInputs("uploadForm");
        document.getElementById("uploadResult").innerHTML = "";
        uploadDatasetSection.style.display = "none";
        document.getElementById("dataPreviewSection").style.display = "none";
        recommendationSection.style.display = "none";
        configSection.style.display = "none";
        predictionSection.style.display = "none";
        exportSection.style.display = "none";
        classImbalanceSection.style.display = "none";
        document.getElementById("dynamicInputs").innerHTML = "";
        document.getElementById("classDistributionResult").innerHTML = "";
        document.getElementById("smoteOption").style.display = "none";
        loadModelSection.style.display = "block";
        resetFileInputs("loadModelForm");
        document.getElementById("loadModelResult").innerHTML = "";
        loadedFeaturesSection.style.display = "none";
      } else if (notTrainedOption.checked) {
        resetFileInputs("loadModelForm");
        document.getElementById("loadModelResult").innerHTML = "";
        loadModelSection.style.display = "none";
        loadedFeaturesSection.style.display = "none";
        uploadDatasetSection.style.display = "block";
      }
    }
    document.getElementById("optionTrained").addEventListener("change", toggleSections);
    document.getElementById("optionNotTrained").addEventListener("change", toggleSections);
    toggleSections();

    // Handle file input change for dataset upload
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("dataset");
    fileInput.addEventListener('change', function(e) {
      const placeholder = this.parentNode.querySelector('.file-input-placeholder');
      const files = e.target.files;
      if (files.length === 0) {
        placeholder.textContent = 'Choose files...';
      } else {
        const fileNames = Array.from(files).map(f => f.name);
        placeholder.textContent = `${files.length} file(s) selected: ${fileNames.join(', ')}`;
      }
    });

    uploadForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const uploadResult = document.getElementById("uploadResult");
      const files = fileInput.files;
      if (files.length === 0) {
        uploadResult.innerHTML = "Error: No files selected.";
        return;
      }
      uploadResult.innerHTML = "Uploading and reading the file(s)...";
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append("dataset", file);
      });
      fetch("/upload", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(result => {
        if (result.columns) {
          const profile = result.dataset.profile;
          let html = `<p>${escapeHtml(result.message)}</p>`;
          html += `<p>Detected dataset: <strong>${escapeHtml(profile.name)}</strong> (${escapeHtml(profile.variant)}); label column: <strong>${escapeHtml(profile.label_column || "not detected")}</strong>; split: ${escapeHtml(profile.split_strategy)}.</p>`;
          if (profile.blocked_columns.length > 0) {
            html += `<p>Never used as features (label-derived, IDs or time): ${escapeHtml(profile.blocked_columns.join(", "))}</p>`;
          }
          uploadResult.innerHTML = html;

          populateColumns(result.columns, result.dataset);

          // Reset balancing method option
          balancingMethodSelect.value = "none";
          document.getElementById("smoteOption").style.display = "none";
          document.getElementById("classDistributionResult").innerHTML = "";
          document.getElementById("balancingPreviewResult").innerHTML = "";

          // Reset target value filter section
          document.getElementById("targetValueFilterSection").style.display = "none";
          document.getElementById("targetFilterResult").innerHTML = "";
          document.getElementById("targetValuesToRemove").innerHTML = "";

          // Show data preview section and fetch preview
          document.getElementById("dataPreviewSection").style.display = "block";
          fetchDataPreview();

          // Show all sections
          document.getElementById("classImbalanceSection").style.display = "block";
          document.getElementById("recommendationSection").style.display = "block";
          document.getElementById("configSection").style.display = "block";

          loadAvailableModels();
        } else {
          uploadResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        uploadResult.textContent = "Error: " + error;
      });
    });

    // Handle file input change for model file upload
    const modelFileInput = document.getElementById('modelFile');
    const modelFilePlaceholder = document.getElementById('modelFilePlaceholder');
    modelFileInput.addEventListener('change', function(e) {
      const files = e.target.files;
      if (files.length === 0) {
        modelFilePlaceholder.textContent = 'Choose file...';
      } else {
        const fileNames = Array.from(files).map(f => f.name);
        modelFilePlaceholder.textContent = `${files.length} file(s) selected: ${fileNames.join(', ')}`;
      }
    });

    // Load model handler: only signed bundles from this install are accepted
    const loadModelForm = document.getElementById("loadModelForm");
    loadModelForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const loadResult = document.getElementById("loadModelResult");
      if (modelFileInput.files.length === 0) {
        loadResult.innerHTML = "Error: Please select a model file.";
        return;
      }
      loadResult.innerHTML = "Checking and loading the bundle...";
      const formData = new FormData();
      formData.append("model_file", modelFileInput.files[0]);
      fetch("/load_model", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(result => {
        if (result.features_info) {
          let accHTML = `<p>${escapeHtml(result.message)}</p>`;
          accHTML += `<p>Trained on ${escapeHtml(result.dataset || "unknown data")} at ${escapeHtml(result.created_at)}.</p>`;
          accHTML += "<table class='metrics-table'><thead><tr><th>Model</th><th>CV macro-F1</th><th>Test macro-F1</th></tr></thead><tbody>";
          result.models.forEach(model => {
            accHTML += `<tr><td>${escapeHtml(model.name)}${escapeHtml(model.id === result.selected ? " (selected)" : "")}</td><td>${escapeHtml(fmt(model.cv_macro_f1, 3))}</td><td>${escapeHtml(fmt(model.test_macro_f1, 3))}</td></tr>`;
          });
          accHTML += "</tbody></table>";
          result.warnings.forEach(warning => {
            accHTML += `<div class='warning-message'>${escapeHtml(warning)}</div>`;
          });
          loadResult.innerHTML = accHTML;
          const loadedSelect = document.getElementById("loadedAlgorithmSelect");
          loadedSelect.innerHTML = "";
          result.models.forEach(model => {
            const option = document.createElement("option");
            option.value = model.id;
            option.text = model.name;
            option.selected = model.id === result.selected;
            loadedSelect.appendChild(option);
          });
          buildLoadedPredictionForm(result.features_info);
          document.getElementById("loadedFeaturesSection").style.display = "block";
        } else {
          loadResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        loadResult.textContent = "Error: " + error;
      });
    });

    // Feature recommendation handler (training split only, after the leakage guard)
    document.getElementById("recommendBtn").addEventListener("click", function(e) {
      e.preventDefault();
      const recommendationResult = document.getElementById("recommendationResult");
      const kValue = document.getElementById("kValue").value;
      const targetForRec = document.getElementById("targetForRec").value;
      if (!kValue || !targetForRec) {
        recommendationResult.innerHTML = "Please provide a valid k value and select a target.";
        return;
      }
      recommendationResult.innerHTML = "Scoring features on the training split...";
      fetch("/recommend_features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: targetForRec, k: parseInt(kValue), test_size: testSize() })
      })
      .then(response => response.json())
      .then(result => {
        if (result.recommended_features) {
          let recHTML = `<p>Scored on ${escapeHtml(result.rows.toLocaleString())} training rows (${escapeHtml(result.method)}). The recommended features are now ticked in the feature list.</p>`;
          recHTML += "<strong>Recommended Features:</strong><br>";
          const recommendedList = [];
          result.recommended_features.forEach(item => {
            recHTML += `${escapeHtml(item[0])} (Score: ${escapeHtml(fmt(item[1], 2))})<br>`;
            recommendedList.push(item[0]);
          });
          if (result.blocked.length > 0) {
            recHTML += "<br><strong>Blocked by the leakage guard:</strong><br>";
            result.blocked.forEach(item => {
              recHTML += `${escapeHtml(item.column)} (${escapeHtml(item.reason)})<br>`;
            });
          }
          recommendationResult.innerHTML = recHTML;
          setChecked(document.getElementById("inputFeatures"), recommendedList);
        } else {
          recommendationResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        recommendationResult.textContent = "Error: " + error;
      });
    });

    // Class distribution analysis handler
    document.getElementById("analyzeClassBtn").addEventListener("click", function(e) {
      e.preventDefault();
      const classDistributionResult = document.getElementById("classDistributionResult");
      const targetForDist = document.getElementById("targetForDist").value;
      if (!targetForDist) {
        classDistributionResult.innerHTML = "Please select a target column for class distribution analysis.";
        return;
      }
      currentTargetColumn = targetForDist;
      classDistributionResult.innerHTML = "Analyzing class distribution...";
      fetch("/analyze_class_distribution", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: targetForDist, test_size: testSize() })
      })
      .then(response => response.json())
      .then(result => {
        if (result.class_counts) {
          let distHTML = "<strong>Class Distribution (whole dataset):</strong><br>";
          distHTML += `<table><tr><th>Class</th><th>Count</th><th>Percentage</th></tr>`;
          result.sorted_classes.forEach(([className, percentage]) => {
            distHTML += `<tr><td>${escapeHtml(className)}</td><td>${escapeHtml(result.class_counts[className].toLocaleString())}</td><td>${escapeHtml(percentage.toFixed(2))}%</td></tr>`;
          });
          distHTML += `</table>`;
          distHTML += `<p>Total samples: ${escapeHtml(result.total_samples.toLocaleString())}</p>`;
          if (result.test_file_counts) {
            distHTML += `<p>Official test file: ${escapeHtml(Object.entries(result.test_file_counts).map(([c, n]) => c + " " + n).join(", "))}</p>`;
          }

          populateTargetValueFilter(result.class_counts);
          document.getElementById("targetValueFilterSection").style.display = "block";
          document.getElementById("targetFilterResult").innerHTML = "";

          const rec = result.recommendation;
          let ratioDisplay;
          const ratioPercent = rec.imbalance_ratio * 100;
          if (ratioPercent >= 1) {
            ratioDisplay = ratioPercent.toFixed(1) + '%';
          } else if (ratioPercent >= 0.01) {
            ratioDisplay = ratioPercent.toFixed(2) + '%';
          } else if (ratioPercent > 0) {
            ratioDisplay = ratioPercent.toExponential(2) + ' (1:' + Math.round(1 / rec.imbalance_ratio) + ')';
          } else {
            ratioDisplay = '0%';
          }
          distHTML += `<div class='recommendation-box'>`;
          distHTML += `<h4>📊 Training split analysis</h4>`;
          distHTML += `<p class='section-description'>Computed on the ${escapeHtml(rec.train_rows.toLocaleString())} training rows only (${escapeHtml(result.split.strategy)} split); the test rows never influence this choice.</p>`;
          distHTML += `<div class='analysis-grid'>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Imbalance Ratio:</span> <span class='analysis-value'>${escapeHtml(ratioDisplay)}</span></div>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Severity:</span> <span class='analysis-value severity-${escapeHtml(rec.severity)}'>${escapeHtml(rec.severity.toUpperCase())}</span></div>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Training rows:</span> <span class='analysis-value'>${escapeHtml(rec.train_rows.toLocaleString())} (${escapeHtml(rec.size)})</span></div>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Features:</span> <span class='analysis-value'>${escapeHtml(rec.features)} (${escapeHtml(rec.dimensionality)} dimensionality)</span></div>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Classes:</span> <span class='analysis-value'>${escapeHtml(rec.classes)} ${escapeHtml(rec.classes > 2 ? '(multi-class)' : '(binary)')}</span></div>`;
          distHTML += `<div class='analysis-item'><span class='analysis-label'>Smallest Class:</span> <span class='analysis-value ${escapeHtml(rec.minority_count < 6 ? "severity-extreme" : "")}'>${escapeHtml(rec.minority_count.toLocaleString())} samples</span></div>`;
          distHTML += `</div>`;
          distHTML += `<div class='recommendation-section'>`;
          distHTML += `<h4>✅ Recommended Method: <span class='recommended-method'>${escapeHtml(BALANCING_NAMES[rec.recommended] || rec.recommended)}</span></h4>`;
          distHTML += `<p class='recommendation-reason'>${escapeHtml(rec.reason)}</p>`;
          if (rec.alternatives && rec.alternatives.length > 0) {
            distHTML += `<p class='alternatives'><strong>Alternatives:</strong> ${escapeHtml(rec.alternatives.map(m => BALANCING_NAMES[m] || m).join(', '))}</p>`;
          }
          distHTML += `</div></div>`;

          document.getElementById("smoteOption").style.display = "block";
          balancingMethodSelect.value = rec.recommended;
          balancingHint.innerHTML = BALANCING_HINTS[rec.recommended] || "";
          classDistributionResult.innerHTML = distHTML;
        } else {
          classDistributionResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        classDistributionResult.textContent = "Error: " + error;
      });
    });

    // Training results: split, leakage guard, leaderboard (CV on train + test scored once)
    function renderTrainingResult(result) {
      const split = result.split;
      let configHTML = `<p>${escapeHtml(result.message)}</p>`;
      configHTML += `<div class='info-message'>Split: ${escapeHtml(split.strategy)} — ${escapeHtml(split.train_rows.toLocaleString())} training rows, ${escapeHtml(split.test_rows.toLocaleString())} test rows`;
      if (split.dedupe && split.dedupe.removed > 0) {
        configHTML += `; ${escapeHtml(split.dedupe.removed.toLocaleString())} duplicate rows removed before splitting`;
      }
      configHTML += `. Cross-validation: ${escapeHtml(result.cv.folds)} folds on ${escapeHtml(result.cv.rows.toLocaleString())} training rows${escapeHtml(result.cv.sampled ? " (sample)" : "")}.</div>`;
      if (split.reason_codes.length > 0) {
        configHTML += `<div class='warning-message'>The ${escapeHtml(split.requested)} split could not be used: ${escapeHtml(split.reason_codes.map(splitMessage).join(" "))}</div>`;
      }
      split.warnings.forEach(code => {
        configHTML += `<div class='warning-message' title='${escapeHtml(code)}'>${escapeHtml(splitMessage(code))}</div>`;
      });
      if (result.leakage.blocked.length > 0) {
        configHTML += `<div class='warning-message'>Left out by the leakage guard: ${escapeHtml(result.leakage.blocked.map(b => b.column + " (" + b.reason + ")").join(", "))}</div>`;
      }
      configHTML += `<p>Balancing: ${escapeHtml(BALANCING_NAMES[result.balancing.method] || result.balancing.method)}${escapeHtml(result.balancing.requested === "recommended" ? " (recommended from the training split)" : "")}.</p>`;
      configHTML += "<strong>Leaderboard</strong> (ranked by CV macro-F1; ★ = selected):<br>";
      configHTML += "<div class='table-container'><table class='metrics-table'><thead><tr><th>Model</th><th>CV macro-F1</th><th>Test macro-F1</th><th>Test balanced acc.</th><th>Test accuracy</th><th>Weakest class recall</th><th>Fit time</th></tr></thead><tbody>";
      result.leaderboard.forEach(row => {
        configHTML += `<tr><td>${escapeHtml(row.selected ? "★ " : "")}${escapeHtml(row.title)}</td><td>${escapeHtml(fmt(row.cv_macro_f1_mean, 3))} ± ${escapeHtml(fmt(row.cv_macro_f1_std, 3))}</td><td>${escapeHtml(fmt(row.test_macro_f1, 3))}</td><td>${escapeHtml(fmt(row.test_balanced_accuracy, 3))}</td><td>${escapeHtml(fmt(row.test_accuracy, 3))}</td><td>${escapeHtml(fmt(row.test_min_recall, 3))}</td><td>${escapeHtml(fmt(row.fit_seconds, 1))} s</td></tr>`;
      });
      configHTML += "</tbody></table></div>";
      result.failures.forEach(failure => {
        configHTML += `<div class='warning-message'>${escapeHtml(modelTitle(failure.name))} failed and was left out.</div>`;
      });
      result.skipped.forEach(([name, reason]) => {
        configHTML += `<div class='warning-message'>${escapeHtml(modelTitle(name))} skipped: ${escapeHtml(reason)}</div>`;
      });
      if (result.classes_only_in_test.length > 0) {
        configHTML += `<div class='warning-message'>Classes only in the test split (no model can predict them): ${escapeHtml(result.classes_only_in_test.join(", "))}</div>`;
      }
      return configHTML;
    }

    // Configuration and training handler
    const configForm = document.getElementById("configForm");
    configForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const configResult = document.getElementById("configResult");
      const selectedTarget = document.getElementById("targetColumn").value;
      const selectedInputs = checkedValues(document.getElementById("inputFeatures")).filter(f => f !== selectedTarget);
      const selectedModels = checkedValues(document.getElementById("modelSelection"));

      if (selectedInputs.length === 0) {
        configResult.innerHTML = "<span style='color: #dc3545;'>❌ Please tick at least one input feature.</span>";
        return;
      }
      if (selectedModels.length === 0) {
        configResult.innerHTML = "<span style='color: #dc3545;'>❌ Please tick at least one model to train.</span>";
        return;
      }
      configResult.innerHTML = "Training models (cross-validation on the training split, then one test)... this can take a while on big datasets.";

      // One training at a time (the server also answers 409 to a second request)
      const trainButton = configForm.querySelector('button[type="submit"]');
      trainButton.disabled = true;
      fetch("/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_features: selectedInputs,
          target: selectedTarget,
          balancing_method: balancingMethodSelect.value,
          test_size: testSize(),
          selected_models: selectedModels
        })
      })
      .then(response => response.json())
      .then(result => {
        if (result.leaderboard) {
          configResult.innerHTML = renderTrainingResult(result);

          confusionMatrixData = result.confusion_matrices;
          const cmModelSelect = document.getElementById("cmModelSelect");
          const algorithmSelect = document.getElementById("algorithmSelect");
          cmModelSelect.innerHTML = "";
          algorithmSelect.innerHTML = "";
          result.leaderboard.forEach(row => {
            const cmOption = document.createElement("option");
            cmOption.value = row.model;
            cmOption.text = row.title;
            cmOption.selected = row.selected;
            cmModelSelect.appendChild(cmOption);
            const option = document.createElement("option");
            option.value = row.model;
            option.text = `${row.title} (CV macro-F1 ${fmt(row.cv_macro_f1_mean, 3)})`;
            option.selected = row.selected;
            algorithmSelect.appendChild(option);
          });
          document.getElementById("confusionMatrixSection").style.display = "block";
          const chosen = confusionMatrixData[result.selected];
          if (chosen) {
            displayConfusionMatrix(chosen.matrix, chosen.labels, result.selected);
          }
          buildPredictionForm(result.features_info);
          document.getElementById("predictionSection").style.display = "block";
          document.getElementById("exportSection").style.display = "block";
          document.getElementById("exportResult").innerHTML = "";
        } else {
          configResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        configResult.textContent = "Error: " + error;
      })
      .finally(() => {
        trainButton.disabled = false;
      });
    });

    // Clear workspace: the server forgets this session's dataset and models
    const clearWorkspaceBtn = document.getElementById("clearWorkspaceBtn");
    clearWorkspaceBtn.addEventListener("click", function() {
      if (!window.confirm("Remove the uploaded dataset and trained models from this session?")) {
        return;
      }
      const clearWorkspaceResult = document.getElementById("clearWorkspaceResult");
      fetch("/workspace", { method: "DELETE" })
      .then(response => {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        window.location.reload();
      })
      .catch(error => {
        clearWorkspaceResult.textContent = "Error: " + error.message;
      });
    });

    // Export model handler: a signed bundle, then a download link
    const exportForm = document.getElementById("exportForm");
    exportForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const exportResult = document.getElementById("exportResult");
      exportResult.textContent = "Saving the signed bundle...";
      const modelName = document.getElementById("modelName").value;
      fetch("/export_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      })
      .then(response => response.json())
      .then(result => {
        if (result.download_url) {
          exportResult.innerHTML = "";
          const message = document.createElement("p");
          message.textContent = result.message;
          const details = document.createElement("p");
          details.textContent = "Run " + result.run_id + ", SHA-256 " + result.sha256.substring(0, 16) + "…";
          const link = document.createElement("a");
          link.href = result.download_url;
          link.download = result.download_name;
          link.className = "btn btn-sm";
          link.textContent = "Download " + result.download_name;
          exportResult.appendChild(message);
          exportResult.appendChild(details);
          exportResult.appendChild(link);
        } else {
          exportResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        exportResult.textContent = "Error: " + error;
      });
    });

    // Build prediction form for session model
    function buildPredictionForm(featuresInfo) {
      const dynamicDiv = document.getElementById("dynamicInputs");
      dynamicDiv.innerHTML = "";
      featuresInfo.forEach((feature, index) => {
        const group = document.createElement("div");
        group.className = "form-group";
        const label = document.createElement("label");
        label.textContent = feature.name + ":";
        let input;
        if (feature.type === "numeric") {
          input = document.createElement("input");
          input.type = "number";
          input.step = "any";
        } else if (feature.type === "categorical") {
          input = document.createElement("select");
          feature.classes.forEach(cls => {
            const option = document.createElement("option");
            option.value = cls;
            option.text = cls;
            input.appendChild(option);
          });
        } else {
          input = document.createElement("input");
          input.type = "text";
        }
        input.id = "session-feature-" + index;
        input.name = feature.name;
        input.required = true;
        label.htmlFor = input.id;
        group.appendChild(label);
        group.appendChild(input);
        dynamicDiv.appendChild(group);
      });
    }

    // Build prediction form for loaded model
    function buildLoadedPredictionForm(featuresInfo) {
      const dynamicDiv = document.getElementById("loadedDynamicInputs");
      dynamicDiv.innerHTML = "";
      featuresInfo.forEach((feature, index) => {
        const group = document.createElement("div");
        group.className = "form-group";
        const label = document.createElement("label");
        label.textContent = feature.name + ":";
        let input;
        if (feature.type === "numeric") {
          input = document.createElement("input");
          input.type = "number";
          input.step = "any";
        } else if (feature.type === "categorical") {
          input = document.createElement("select");
          feature.classes.forEach(cls => {
            const option = document.createElement("option");
            option.value = cls;
            option.text = cls;
            input.appendChild(option);
          });
        } else {
          input = document.createElement("input");
          input.type = "text";
        }
        input.id = "loaded-feature-" + index;
        input.name = feature.name;
        input.required = true;
        label.htmlFor = input.id;
        group.appendChild(label);
        group.appendChild(input);
        dynamicDiv.appendChild(group);
      });
    }

    // Values of a prediction form: feature name -> entered text
    function formValues(containerId) {
      const values = {};
      document.getElementById(containerId).querySelectorAll("input, select").forEach(input => {
        values[input.name] = input.value;
      });
      return values;
    }

    // Prediction handler for session model
    const predictionForm = document.getElementById("predictionForm");
    predictionForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const predictionResult = document.getElementById("predictionResult");
      predictionResult.textContent = "Predicting...";
      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: formValues("dynamicInputs"),
                               algorithm: document.getElementById("algorithmSelect").value })
      })
      .then(response => response.json())
      .then(result => {
        if (result.prediction !== undefined) {
          predictionResult.textContent = "Predicted Value: " + result.prediction;
        } else {
          predictionResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        predictionResult.textContent = "Error: " + error;
      });
    });

    // Prediction handler for loaded model
    const predictLoadedForm = document.getElementById("predictLoadedForm");
    predictLoadedForm.addEventListener("submit", function(e) {
      e.preventDefault();
      const loadedPredictionResult = document.getElementById("loadedPredictionResult");
      loadedPredictionResult.textContent = "Predicting...";
      fetch("/predict_loaded", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: formValues("loadedDynamicInputs"),
                               algorithm: document.getElementById("loadedAlgorithmSelect").value })
      })
      .then(response => response.json())
      .then(result => {
        if (result.prediction !== undefined) {
          loadedPredictionResult.textContent = "Predicted Value: " + result.prediction;
        } else {
          loadedPredictionResult.textContent = "Error: " + result.error;
        }
      })
      .catch(error => {
        loadedPredictionResult.textContent = "Error: " + error;
      });
    });
  } // End Dynamic IDS Section
});
