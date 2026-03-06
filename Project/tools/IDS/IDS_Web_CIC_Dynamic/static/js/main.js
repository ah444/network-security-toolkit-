// Define global variable for loaded model configuration
var loaded_model_config = {};

// Store confusion matrix data globally
var confusionMatrixData = {};
var classLabels = [];

document.addEventListener("DOMContentLoaded", function() {

  /********** DYNAMIC IDS FUNCTIONALITY **********/
  if (document.getElementById("uploadForm")) {
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
              <div class="stat-item"><strong>Total Rows:</strong> ${stats.total_rows.toLocaleString()}</div>
              <div class="stat-item"><strong>Total Columns:</strong> ${stats.total_columns}</div>
              <div class="stat-item"><strong>Numeric Columns:</strong> ${stats.numeric_columns}</div>
              <div class="stat-item"><strong>Categorical Columns:</strong> ${stats.categorical_columns}</div>
            </div>
          `;
          
          // Build preview table
          let tableHTML = '<table class="preview-table"><thead><tr>';
          result.columns.forEach(col => {
            tableHTML += `<th>${col}</th>`;
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
              tableHTML += `<td>${cellValue !== null && cellValue !== undefined ? cellValue : ''}</td>`;
            });
            tableHTML += '</tr>';
          });
          
          tableHTML += '</tbody></table>';
          dataPreviewTable.innerHTML = tableHTML;
        } else {
          dataStats.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        dataStats.innerHTML = "Error loading preview: " + error;
      });
    }

    // Refresh preview button handler
    document.getElementById("refreshPreviewBtn").addEventListener("click", fetchDataPreview);

    // Helper function to populate all column dropdowns
    function populateColumnDropdowns(columns) {
      const inputSelect = document.getElementById("inputFeatures");
      const targetSelect = document.getElementById("targetColumn");
      const targetRecSelect = document.getElementById("targetForRec");
      const targetDistSelect = document.getElementById("targetForDist");
      
      inputSelect.innerHTML = "";
      targetSelect.innerHTML = "";
      targetRecSelect.innerHTML = "";
      targetDistSelect.innerHTML = "";
      
      columns.forEach(col => {
        const opt1 = document.createElement("option");
        opt1.value = col;
        opt1.text = col;
        inputSelect.appendChild(opt1);

        const opt2 = document.createElement("option");
        opt2.value = col;
        opt2.text = col;
        targetSelect.appendChild(opt2);

        const opt3 = document.createElement("option");
        opt3.value = col;
        opt3.text = col;
        targetRecSelect.appendChild(opt3);
        
        const opt4 = document.createElement("option");
        opt4.value = col;
        opt4.text = col;
        targetDistSelect.appendChild(opt4);
      });
    }

    // Store current target column for filtering
    var currentTargetColumn = null;

    // Function to populate target value filter dropdown
    function populateTargetValueFilter(classCounts) {
      const select = document.getElementById("targetValuesToRemove");
      select.innerHTML = "";
      
      // Sort by count (ascending) so small classes appear first
      const sortedClasses = Object.entries(classCounts).sort((a, b) => a[1] - b[1]);
      
      sortedClasses.forEach(([className, count]) => {
        const opt = document.createElement("option");
        opt.value = className;
        opt.text = `${className} (${count.toLocaleString()} samples)`;
        select.appendChild(opt);
      });
    }

    // Remove target values handler
    document.getElementById("removeTargetValuesBtn").addEventListener("click", function() {
      const select = document.getElementById("targetValuesToRemove");
      const selectedValues = Array.from(select.selectedOptions).map(opt => opt.value);
      const resultDiv = document.getElementById("targetFilterResult");
      
      if (selectedValues.length === 0) {
        resultDiv.innerHTML = "<span style='color: #ffc107;'>⚠️ Please select at least one value to remove, or click 'Skip' to keep all values.</span>";
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
          resultDiv.innerHTML = `<span style='color: #28a745;'>✅ ${result.message}</span>`;
          
          // Update the target value filter dropdown with remaining values
          populateTargetValueFilter(result.new_class_counts);
          
          // Refresh data preview
          fetchDataPreview();
          
          // Re-trigger class analysis to update the distribution display
          document.getElementById("analyzeClassBtn").click();
        } else {
          resultDiv.innerHTML = `<span style='color: #dc3545;'>❌ Error: ${result.error}</span>`;
        }
      })
      .catch(error => {
        resultDiv.innerHTML = `<span style='color: #dc3545;'>❌ Error: ${error}</span>`;
      });
    });

    // Skip target filter handler
    document.getElementById("skipTargetFilterBtn").addEventListener("click", function() {
      const resultDiv = document.getElementById("targetFilterResult");
      resultDiv.innerHTML = "<span style='color: #17a2b8;'>ℹ️ Keeping all target values.</span>";
      
      // Show balancing options
      document.getElementById("smoteOption").style.display = "block";
    });

    // Train/Test Split Slider Handler
    const trainTestSlider = document.getElementById("trainTestSlider");
    const splitLabel = document.getElementById("splitLabel");
    
    trainTestSlider.addEventListener("input", function() {
      const testValSize = parseInt(this.value);
      const trainSize = 100 - testValSize;
      const valSize = Math.round(testValSize / 2);
      const testSize = testValSize - valSize;
      splitLabel.textContent = `Train: ${trainSize}% | Val: ${valSize}% | Test: ${testSize}%`;
    });

    // Balancing Method Hint Handler
    const balancingMethodSelect = document.getElementById("balancingMethod");
    const balancingHint = document.getElementById("balancingHint");
    
    const balancingHints = {
      "none": "No balancing will be applied. Use this if your dataset is already balanced.",
      "class_weight": "<strong>Recommended!</strong> Fast and memory-efficient. Adjusts model weights to give more importance to minority classes.",
      "smote": "Creates synthetic samples for minority classes. Good for continuous features. May increase training time.",
      "smote_enn": "SMOTE + Edited Nearest Neighbors. Creates synthetic samples then removes noisy ones. Best for cleaner decision boundaries.",
      "adasyn": "Adaptive SMOTE - focuses on harder-to-learn samples. Better than SMOTE for complex boundaries.",
      "borderline_smote": "Only generates synthetic samples near class boundaries. Good when classes overlap.",
      "random_undersample": "Removes random samples from majority class. Very fast but may lose important data.",
      "tomek": "Removes overlapping samples between classes. Cleans decision boundaries without adding data.",
      "smote_tomek": "SMOTE + Tomek Links. Creates synthetic samples then cleans boundaries. Good all-around choice."
    };
    
    if (balancingMethodSelect && balancingHint) {
      balancingMethodSelect.addEventListener("change", function() {
        balancingHint.innerHTML = balancingHints[this.value] || "Select a method to handle class imbalance.";
      });
    }

    // Confusion Matrix Display Function
    function displayConfusionMatrix(matrix, labels, modelName, datasetType) {
      const container = document.getElementById("confusionMatrixDisplay");
      
      if (!matrix || matrix.length === 0) {
        container.innerHTML = "<p>No confusion matrix data available.</p>";
        return;
      }
      
      let html = `<div class="cm-header"><strong>${modelName}</strong> - ${datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Set</div>`;
      html += '<table class="confusion-matrix">';
      
      // Header row with predicted labels
      html += '<tr><th></th><th colspan="' + matrix.length + '">Predicted</th></tr>';
      html += '<tr><th></th>';
      
      // Use labels if available, otherwise use indices
      const displayLabels = labels && labels.length === matrix.length ? labels : 
                           matrix.map((_, i) => `Class ${i}`);
      
      displayLabels.forEach(label => {
        // Truncate long labels
        const truncLabel = typeof label === 'string' && label.length > 12 ? 
                          label.substring(0, 10) + '...' : label;
        html += `<th title="${label}">${truncLabel}</th>`;
      });
      html += '</tr>';
      
      // Data rows
      matrix.forEach((row, i) => {
        html += '<tr>';
        const rowLabel = displayLabels[i];
        const truncLabel = typeof rowLabel === 'string' && rowLabel.length > 12 ? 
                          rowLabel.substring(0, 10) + '...' : rowLabel;
        html += `<th title="${rowLabel}">${truncLabel}</th>`;
        
        // Calculate row total for percentage
        const rowTotal = row.reduce((sum, val) => sum + val, 0);
        
        row.forEach((val, j) => {
          // Highlight diagonal (correct predictions)
          const isCorrect = i === j;
          const percentage = rowTotal > 0 ? ((val / rowTotal) * 100).toFixed(1) : 0;
          const bgColor = isCorrect ? 
            `rgba(0, 128, 0, ${Math.min(val / rowTotal, 0.8)})` : 
            `rgba(255, 0, 0, ${Math.min(val / rowTotal, 0.5)})`;
          html += `<td style="background-color: ${bgColor}; color: ${val > rowTotal * 0.5 ? 'white' : 'black'}" title="${val} (${percentage}%)">${val}</td>`;
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
      const modelSelect = document.getElementById("cmModelSelect");
      const datasetType = document.getElementById("cmDatasetType").value;
      const modelName = modelSelect.value;
      
      if (!modelName) {
        document.getElementById("confusionMatrixDisplay").innerHTML = "<p>Please select a model.</p>";
        return;
      }
      
      if (confusionMatrixData[datasetType] && confusionMatrixData[datasetType][modelName]) {
        displayConfusionMatrix(confusionMatrixData[datasetType][modelName], classLabels, modelName, datasetType);
      } else {
        // Fetch from server if not cached
        fetch("/get_confusion_matrix", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: modelName, dataset_type: datasetType })
        })
        .then(response => response.json())
        .then(result => {
          if (result.confusion_matrix) {
            displayConfusionMatrix(result.confusion_matrix, result.class_labels, modelName, datasetType);
          } else {
            document.getElementById("confusionMatrixDisplay").innerHTML = "Error: " + result.error;
          }
        })
        .catch(error => {
          document.getElementById("confusionMatrixDisplay").innerHTML = "Error: " + error;
        });
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
      console.log("Upload form submitted");
      const uploadResult = document.getElementById("uploadResult");
      const files = fileInput.files;
      if (files.length === 0) {
        uploadResult.innerHTML = "Error: No files selected.";
        return;
      }
      uploadResult.innerHTML = "Uploading files...";
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
        console.log("Upload result:", result);
        if (result.columns) {
          uploadResult.innerHTML = "File(s) uploaded successfully!";
          
          // Populate all column dropdowns
          populateColumnDropdowns(result.columns);
          
          // Reset balancing method option
          document.getElementById("balancingMethod").value = "none";
          document.getElementById("smoteOption").style.display = "none";
          document.getElementById("classDistributionResult").innerHTML = "";
          
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
        } else {
          uploadResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        uploadResult.innerHTML = "Error: " + error;
        console.error("Upload fetch error:", error);
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

    // Load model handler
    const loadModelForm = document.getElementById("loadModelForm");
    loadModelForm.addEventListener("submit", function(e) {
      e.preventDefault();
      console.log("Load Model form submitted");
      const loadResult = document.getElementById("loadModelResult");
      if (modelFileInput.files.length === 0) {
        loadResult.innerHTML = "Error: Please select a model file.";
        return;
      }
      loadResult.innerHTML = "Loading model...";
      const formData = new FormData();
      formData.append("model_file", modelFileInput.files[0]);
      fetch("/load_model", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(result => {
        console.log("Loaded model result:", result);
        if (result.message) {
          let loadedKey = "";
          var validationKeys = Object.keys(result.accuracies.Validation || {});
          if (validationKeys.length === 1) {
            loadedKey = validationKeys[0];
          } else if (validationKeys.length > 1) {
            loadedKey = validationKeys[0];
          }
          let accHTML = "<strong>Accuracies:</strong><br>";
          if (result.accuracies.Validation && result.accuracies.Validation[loadedKey]) {
            accHTML += `<u>Validation - ${loadedKey}:</u> ${result.accuracies.Validation[loadedKey]}<br>`;
          }
          if (result.accuracies.Test && result.accuracies.Test[loadedKey]) {
            accHTML += `<u>Test - ${loadedKey}:</u> ${result.accuracies.Test[loadedKey]}<br>`;
          }
          loadResult.innerHTML = result.message + "<br>" + accHTML;
          buildLoadedPredictionForm(result.features_info);
          document.getElementById("loadedFeaturesSection").style.display = "block";
        } else {
          loadResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        loadResult.innerHTML = "Error: " + error;
        console.error("Load model fetch error:", error);
      });
    });

    // Feature recommendation handler
    document.getElementById("recommendBtn").addEventListener("click", function(e) {
      e.preventDefault();
      const recommendationResult = document.getElementById("recommendationResult");
      const kValue = document.getElementById("kValue").value;
      const targetForRec = document.getElementById("targetForRec").value;
      if (!kValue || !targetForRec) {
        recommendationResult.innerHTML = "Please provide a valid k value and select a target.";
        return;
      }
      recommendationResult.innerHTML = "Fetching recommendations...";
      fetch("/recommend_features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: targetForRec, k: parseInt(kValue) })
      })
      .then(response => response.json())
      .then(result => {
        if (result.recommended_features) {
          let recHTML = "";
          
          // Show dataset status (original or balanced)
          if (result.using_balanced_dataset) {
            recHTML += "<div class='success-message'><strong>Using SMOTE-balanced dataset</strong> for feature recommendation.</div><br>";
          }
          
          recHTML += "<strong>Recommended Features:</strong><br>";
          const recommendedList = [];
          result.recommended_features.forEach(item => {
            recHTML += `${item[0]} (Score: ${item[1].toFixed(2)})<br>`;
            recommendedList.push(item[0]);
          });
          recommendationResult.innerHTML = recHTML;
          const inputSelect = document.getElementById("inputFeatures");
          for (let i = 0; i < inputSelect.options.length; i++) {
            const opt = inputSelect.options[i];
            if (recommendedList.includes(opt.value)) {
              opt.selected = true;
            }
          }
        } else {
          recommendationResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        recommendationResult.innerHTML = "Error: " + error;
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
      
      // Store current target column for filtering
      currentTargetColumn = targetForDist;
      
      classDistributionResult.innerHTML = "Analyzing class distribution...";
      fetch("/analyze_class_distribution", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: targetForDist })
      })
      .then(response => response.json())
      .then(result => {
        if (result.class_counts) {
          let distHTML = "";
          
          // Show dataset status (original or balanced)
          if (result.using_balanced_dataset) {
            distHTML += "<div class='success-message'><strong>Using SMOTE-balanced dataset</strong> for class distribution analysis.</div><br>";
          }
          
          distHTML += "<strong>Class Distribution:</strong><br>";
          distHTML += `<table><tr><th>Class</th><th>Count</th><th>Percentage</th></tr>`;
          result.sorted_classes.forEach(([className, percentage]) => {
            distHTML += `<tr><td>${className}</td><td>${result.class_counts[className].toLocaleString()}</td><td>${percentage.toFixed(2)}%</td></tr>`;
          });
          distHTML += `</table>`;
          distHTML += `<p>Total samples: ${result.total_samples.toLocaleString()}</p>`;
          
          // Populate target value filter dropdown
          populateTargetValueFilter(result.class_counts);
          
          // Show target value filter section
          document.getElementById("targetValueFilterSection").style.display = "block";
          document.getElementById("targetFilterResult").innerHTML = "";
          
          // Show recommendation if available
          if (result.recommendation) {
            const rec = result.recommendation;
            const analysis = rec.analysis;
            
            // Method display names
            const methodNames = {
              "none": "None (No Balancing)",
              "class_weight": "Class Weights",
              "smote": "SMOTE",
              "smote_enn": "SMOTE-ENN",
              "adasyn": "ADASYN",
              "borderline_smote": "Borderline-SMOTE",
              "random_undersample": "Random Undersampling",
              "tomek": "Tomek Links",
              "smote_tomek": "SMOTE + Tomek Links"
            };
            
            // Build analysis box
            distHTML += `<div class='recommendation-box'>`;
            distHTML += `<h4>📊 Data Analysis</h4>`;
            distHTML += `<div class='analysis-grid'>`;
            // Format imbalance ratio with appropriate precision
            let ratioDisplay;
            const ratioPercent = analysis.imbalance_ratio * 100;
            if (ratioPercent >= 1) {
              ratioDisplay = ratioPercent.toFixed(1) + '%';
            } else if (ratioPercent >= 0.01) {
              ratioDisplay = ratioPercent.toFixed(2) + '%';
            } else if (ratioPercent > 0) {
              ratioDisplay = ratioPercent.toExponential(2) + ' (1:' + Math.round(1/analysis.imbalance_ratio) + ')';
            } else {
              ratioDisplay = '0%';
            }
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Imbalance Ratio:</span> <span class='analysis-value'>${ratioDisplay}</span></div>`;
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Severity:</span> <span class='analysis-value severity-${analysis.imbalance_severity}'>${analysis.imbalance_severity.toUpperCase()}</span></div>`;
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Dataset Size:</span> <span class='analysis-value'>${analysis.total_samples.toLocaleString()} (${analysis.dataset_size})</span></div>`;
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Features:</span> <span class='analysis-value'>${analysis.num_features} (${analysis.dimensionality} dimensionality)</span></div>`;
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Classes:</span> <span class='analysis-value'>${analysis.num_classes} ${analysis.is_multiclass ? '(multi-class)' : '(binary)'}</span></div>`;
            distHTML += `<div class='analysis-item'><span class='analysis-label'>Smallest Class:</span> <span class='analysis-value ${analysis.minority_class_count < 6 ? "severity-extreme" : ""}'>${analysis.minority_class_count.toLocaleString()} samples</span></div>`;
            distHTML += `</div>`;
            
            // Show recommendation
            distHTML += `<div class='recommendation-section'>`;
            distHTML += `<h4>✅ Recommended Method: <span class='recommended-method'>${methodNames[rec.recommended] || rec.recommended}</span></h4>`;
            distHTML += `<p class='recommendation-reason'>${rec.reason}</p>`;
            
            // Show alternatives
            if (rec.alternatives && rec.alternatives.length > 0) {
              distHTML += `<p class='alternatives'><strong>Alternatives:</strong> ${rec.alternatives.map(m => methodNames[m] || m).join(', ')}</p>`;
            }
            distHTML += `</div></div>`;
            
            // Show balancing options and auto-select recommended
            document.getElementById("smoteOption").style.display = "block";
            document.getElementById("balancingMethod").value = rec.recommended;
            
            // Update hint
            const balancingHints = {
              "none": "No balancing will be applied. Use original data distribution.",
              "class_weight": "<strong>Recommended!</strong> Fast and memory-efficient. Adjusts model weights to give more importance to minority classes.",
              "smote": "Synthetic Minority Oversampling Technique. Creates synthetic samples by interpolating between existing minority samples.",
              "smote_enn": "SMOTE + Edited Nearest Neighbors. Combines oversampling with cleaning to remove noisy samples.",
              "adasyn": "Adapive Synthetic Sampling. Focuses on harder-to-learn samples by generating more synthetic data near decision boundary.",
              "borderline_smote": "Focuses on samples near the class boundary. More targeted than regular SMOTE.",
              "random_undersample": "Randomly removes majority class samples. Very fast but may lose important information.",
              "tomek": "Removes Tomek links (pairs of very close samples from different classes). Cleans decision boundary.",
              "smote_tomek": "Combines SMOTE oversampling with Tomek links cleaning. Balanced approach."
            };
            document.getElementById("balancingHint").innerHTML = balancingHints[rec.recommended] || "";
            
          } else {
            // Legacy: Show balancing options if imbalanced classes detected
            const imbalanced = result.sorted_classes.some(([_, percentage]) => percentage < 10);
            if (imbalanced && !result.using_balanced_dataset) {
              distHTML += `<p><strong>⚠️ Class imbalance detected!</strong> Select a balancing method below before training.</p>`;
              document.getElementById("smoteOption").style.display = "block";
              document.getElementById("balancingMethod").value = "class_weight";
              document.getElementById("balancingHint").innerHTML = "<strong>Recommended!</strong> Fast and memory-efficient. Adjusts model weights to give more importance to minority classes.";
            } else if (result.using_balanced_dataset) {
              document.getElementById("smoteOption").style.display = "block";
            } else {
              document.getElementById("smoteOption").style.display = "none";
            }
          }
          
          classDistributionResult.innerHTML = distHTML;
        } else {
          classDistributionResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        classDistributionResult.innerHTML = "Error: " + error;
      });
    });

    // Configuration and training handler
    const configForm = document.getElementById("configForm");
    configForm.addEventListener("submit", function(e) {
      e.preventDefault();
      console.log("Config form submitted");
      const configResult = document.getElementById("configResult");
      configResult.innerHTML = "Training models...";
      const inputSelect = document.getElementById("inputFeatures");
      const targetSelect = document.getElementById("targetColumn");
      const selectedInputs = Array.from(inputSelect.selectedOptions).map(opt => opt.value);
      const selectedTarget = targetSelect.value;
      
      // Get balancing method from dropdown
      const balancingMethodSelect = document.getElementById("balancingMethod");
      const balancingMethod = balancingMethodSelect ? balancingMethodSelect.value : "none";
      
      // Get train/test split ratio from slider
      const trainTestSlider = document.getElementById("trainTestSlider");
      const testSize = trainTestSlider ? parseInt(trainTestSlider.value) / 100 : 0.30;
      
      // Log balancing method to console for debugging
      console.log("Balancing Method:", balancingMethod);
      console.log("Test Size:", testSize);
      
      fetch("/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_features: selectedInputs, target: selectedTarget, balancing_method: balancingMethod, test_size: testSize })
      })
      .then(response => response.json())
      .then(result => {
        if (result.message) {
          let configHTML = result.message + "<br><br>";
          
          // Display split information
          if (result.split_info) {
            configHTML += `<div class='info-message'>Data Split: ${result.split_info.split_ratio} (Train: ${result.split_info.train_size} | Val: ${result.split_info.val_size} | Test: ${result.split_info.test_size})</div><br>`;
          }
          
          // Display balancing status
          if (result.balancing_applied && result.balancing_method) {
            const methodNames = {
              "class_weight": "Class Weights",
              "smote": "SMOTE",
              "smote_enn": "SMOTE-ENN",
              "adasyn": "ADASYN",
              "borderline_smote": "Borderline-SMOTE",
              "random_undersample": "Random Undersampling",
              "tomek": "Tomek Links",
              "smote_tomek": "SMOTE + Tomek Links"
            };
            const methodName = methodNames[result.balancing_method] || result.balancing_method;
            configHTML += `<div class='success-message'>${methodName} successfully applied to balance classes!</div><br>`;
          }
          
          // Display Validation Metrics
          configHTML += "<strong>Validation Metrics:</strong><br>";
          configHTML += "<table class='metrics-table'><thead><tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>AUC</th></tr></thead><tbody>";
          Object.entries(result.accuracies.validation).forEach(([algo, acc]) => {
            let algoName = algo === "random_forest" ? "Random Forest" : 
                           algo === "logistic_regression" ? "Logistic Regression" : 
                           algo === "svm" ? "SVM" : algo === "knn" ? "KNN" : algo;
            const f1 = result.f1_scores && result.f1_scores.validation && result.f1_scores.validation[algo];
            const auc = result.auc_scores && result.auc_scores.validation && result.auc_scores.validation[algo];
            configHTML += `<tr>
              <td>${algoName}</td>
              <td>${(acc * 100).toFixed(2)}%</td>
              <td>${f1 !== null && f1 !== undefined ? (f1 * 100).toFixed(2) + '%' : 'N/A'}</td>
              <td>${auc !== null && auc !== undefined ? (auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            </tr>`;
          });
          configHTML += "</tbody></table><br>";
          
          // Display Test Metrics
          configHTML += "<strong>Test Metrics:</strong><br>";
          configHTML += "<table class='metrics-table'><thead><tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>AUC</th></tr></thead><tbody>";
          Object.entries(result.accuracies.test).forEach(([algo, acc]) => {
            let algoName = algo === "random_forest" ? "Random Forest" : 
                           algo === "logistic_regression" ? "Logistic Regression" : 
                           algo === "svm" ? "SVM" : algo === "knn" ? "KNN" : algo;
            const f1 = result.f1_scores && result.f1_scores.test && result.f1_scores.test[algo];
            const auc = result.auc_scores && result.auc_scores.test && result.auc_scores.test[algo];
            configHTML += `<tr>
              <td>${algoName}</td>
              <td>${(acc * 100).toFixed(2)}%</td>
              <td>${f1 !== null && f1 !== undefined ? (f1 * 100).toFixed(2) + '%' : 'N/A'}</td>
              <td>${auc !== null && auc !== undefined ? (auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            </tr>`;
          });
          configHTML += "</tbody></table>";
          configResult.innerHTML = configHTML;
          
          // Store confusion matrix data globally
          if (result.confusion_matrices) {
            confusionMatrixData = result.confusion_matrices;
            classLabels = result.class_labels || [];
            
            // Populate confusion matrix model select
            const cmModelSelect = document.getElementById("cmModelSelect");
            cmModelSelect.innerHTML = "";
            Object.keys(result.accuracies.test).forEach(algo => {
              const option = document.createElement("option");
              option.value = algo;
              let algoName = "";
              if (algo === "random_forest") algoName = "Random Forest";
              else if (algo === "logistic_regression") algoName = "Logistic Regression";
              else if (algo === "svm") algoName = "SVM";
              else if (algo === "knn") algoName = "KNN";
              else algoName = algo;
              option.text = algoName;
              cmModelSelect.appendChild(option);
            });
            
            // Show confusion matrix section
            document.getElementById("confusionMatrixSection").style.display = "block";
            
            // Auto-display first model's confusion matrix
            const firstModel = Object.keys(result.accuracies.test)[0];
            if (firstModel && confusionMatrixData.test && confusionMatrixData.test[firstModel]) {
              displayConfusionMatrix(confusionMatrixData.test[firstModel], classLabels, firstModel, "test");
            }
          }
          
          const algorithmSelect = document.getElementById("algorithmSelect");
          algorithmSelect.innerHTML = "";
          Object.entries(result.accuracies.test).forEach(([algo, acc]) => {
            const option = document.createElement("option");
            option.value = algo;
            let algoName = "";
            if (algo === "random_forest") algoName = "Random Forest";
            else if (algo === "logistic_regression") algoName = "Logistic Regression";
            else if (algo === "svm") algoName = "SVM";
            else if (algo === "knn") algoName = "KNN";
            else algoName = algo;
            option.text = `${algoName} (Test Accuracy: ${(acc * 100).toFixed(2)}%)`;
            algorithmSelect.appendChild(option);
          });
          buildPredictionForm(result.features_info);
          document.getElementById("predictionSection").style.display = "block";
          document.getElementById("exportSection").style.display = "block";
        } else {
          configResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        configResult.innerHTML = "Error: " + error;
      });
    });

    // Export model handler
    const exportForm = document.getElementById("exportForm");
    exportForm.addEventListener("submit", function(e) {
      e.preventDefault();
      console.log("Export form submitted");
      const exportResult = document.getElementById("exportResult");
      exportResult.innerHTML = "Exporting model...";
      const modelName = document.getElementById("modelName").value;
      fetch("/export_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      })
      .then(response => response.json())
      .then(result => {
        if (result.files) {
          exportResult.innerHTML = "Export successful! Files: " + result.files.join(", ");
        } else {
          exportResult.innerHTML = "Error: " + result.error;
        }
      })
      .catch(error => {
        exportResult.innerHTML = "Error: " + error;
      });
    });

    // Build prediction form for session model
    function buildPredictionForm(featuresInfo) {
      const dynamicDiv = document.getElementById("dynamicInputs");
      dynamicDiv.innerHTML = "";
      featuresInfo.forEach(feature => {
        const group = document.createElement("div");
        group.className = "form-group";
        const label = document.createElement("label");
        label.textContent = feature.name + ":";
        group.appendChild(label);
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
        input.id = feature.name;
        input.name = feature.name;
        input.required = true;
        group.appendChild(input);
        dynamicDiv.appendChild(group);
      });
    }

    // Build prediction form for loaded model
    function buildLoadedPredictionForm(featuresInfo) {
      const dynamicDiv = document.getElementById("loadedDynamicInputs");
      dynamicDiv.innerHTML = "";
      featuresInfo.forEach(feature => {
        const group = document.createElement("div");
        group.className = "form-group";
        const label = document.createElement("label");
        label.textContent = feature.name + ":";
        group.appendChild(label);
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
        input.id = feature.name;
        input.name = feature.name;
        input.required = true;
        group.appendChild(input);
        dynamicDiv.appendChild(group);
      });
    }

    // Prediction handler for session model
    const predictionForm = document.getElementById("predictionForm");
    predictionForm.addEventListener("submit", function(e) {
      e.preventDefault();
      console.log("Prediction form submitted");
      const predictionResult = document.getElementById("predictionResult");
      predictionResult.innerHTML = "Predicting...";
      const dynamicDiv = document.getElementById("dynamicInputs");
      const inputs = dynamicDiv.querySelectorAll("input, select");
      const values = {};
      inputs.forEach(input => {
        values[input.name] = input.value;
      });
      const algorithm = document.getElementById("algorithmSelect").value;
      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: values, algorithm: algorithm })
      })
      .then(response => response.json())
      .then(result => {
        if (result.prediction !== undefined) {
          predictionResult.innerHTML = "Predicted Value: " + result.prediction;
          console.log("Prediction result (session model):", result.prediction);
        } else {
          predictionResult.innerHTML = "Error: " + result.error;
          console.error("Session model prediction error:", result.error);
        }
      })
      .catch(error => {
        predictionResult.innerHTML = "Error: " + error;
        console.error("Session model fetch error:", error);
      });
    });

    // Prediction handler for loaded model
    const predictLoadedForm = document.getElementById("predictLoadedForm");
    predictLoadedForm.addEventListener("submit", function(e) {
      e.preventDefault();
      console.log("predictLoadedForm submitted");
      const loadedPredictionResult = document.getElementById("loadedPredictionResult");
      loadedPredictionResult.innerHTML = "Predicting...";
      const dynamicDiv = document.getElementById("loadedDynamicInputs");
      const inputs = dynamicDiv.querySelectorAll("input, select");
      const values = {};
      inputs.forEach(input => {
        values[input.name] = input.value;
      });
      // Convert numeric features if necessary
      const numericFeatures = loaded_model_config.numeric_features || [];
      numericFeatures.forEach(feat => {
        if (values[feat] !== undefined && values[feat] !== "") {
          values[feat] = parseFloat(values[feat]);
        }
      });
      console.log("Predicting with loaded model values:", values);
      fetch("/predict_loaded", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: values })
      })
      .then(response => {
        console.log("Predict loaded response status:", response.status);
        return response.json();
      })
      .then(result => {
        console.log("Predict loaded result:", result);
        if (result.prediction !== undefined) {
          loadedPredictionResult.innerHTML = "Predicted Value: " + result.prediction;
        } else {
          loadedPredictionResult.innerHTML = "Error: " + result.error;
          console.error("Loaded model prediction error:", result.error);
        }
      })
      .catch(error => {
        loadedPredictionResult.innerHTML = "Error: " + error;
        console.error("Loaded model fetch error:", error);
      });
    });
  } // End Dynamic IDS Section
});
