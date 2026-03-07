function runAI() {
    const input = document.getElementById("inputText").value;
    const resultEl = document.getElementById("result");
    const btn = document.getElementById("diagnoseBtn");
    const loading = document.getElementById("loading");

    if (!input) {
        resultEl.textContent = "Please enter symptoms.";
        return;
    }

    // Show spinner, disable button
    loading.classList.add("active");
    btn.disabled = true;
    resultEl.innerHTML = "";

    fetch("/diagnose", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        },
        body: JSON.stringify({ text: input })
    })
        .then(response => response.json())
        .then(data => {

            const outputEl = document.getElementById("output");

            if (data.error) {
                resultEl.textContent = data.error;
                outputEl.classList.add("visible");
                return;
            }

            if (!data.results) {
                resultEl.textContent = data.result;
                outputEl.classList.add("visible");
                return;
            }

            let outputHTML = "";

            data.results.forEach((item, index) => {

                const matchedSymptoms = item.matched_symptoms || [];
                const totalSymptoms = item.total_symptoms != null ? item.total_symptoms : matchedSymptoms.length;

                let symptomsHTML = "";

                if (matchedSymptoms.length > 0) {
                    const tags = matchedSymptoms
                        .map(s => `<span class="symptom-tag">${s}</span>`)
                        .join("");
                    symptomsHTML = `
                        <p class="match-summary">Matched ${matchedSymptoms.length}/${totalSymptoms} symptoms:</p>
                        <div class="symptom-tags">${tags}</div>
                    `;
                }

                outputHTML += `
                    <div class="diagnosis-block">
                        <strong>${index + 1}. ${item.disease}</strong><br>
                        Confidence: ${(item.score * 100).toFixed(2)}%
                        ${symptomsHTML}
                    </div>
                `;
            });

            resultEl.innerHTML = outputHTML;
            document.getElementById("output").classList.add("visible");
            document.getElementById("result-count").textContent = `${data.results.length} result${data.results.length !== 1 ? "s" : ""}`;

        })
        .catch(error => {
            console.error(error);
            resultEl.textContent = "Error connecting to AI.";
        })
        .finally(() => {
            // Always hide spinner and re-enable button
            loading.classList.remove("active");
            btn.disabled = false;
        });
}