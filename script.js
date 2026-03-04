function runAI() {
    const input = document.getElementById("inputText").value;
    const resultEl = document.getElementById("result");

    if (!input) {
        resultEl.textContent = "Please enter symptoms.";
        return;
    }

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

            if (data.error) {
                resultEl.textContent = data.error;
                return;
            }

            if (!data.results) {
                resultEl.textContent = data.result;
                return;
            }

            let outputHTML = "";

            data.results.forEach((item, index) => {
                outputHTML += `
          <strong>${index + 1}. ${item.disease}</strong><br>
          Confidence: ${(item.score * 100).toFixed(2)}%<br><br>
        `;
            });

            resultEl.innerHTML = outputHTML;

        })
        .catch(error => {
            console.error(error);
            resultEl.textContent = "Error connecting to AI.";
        });
}