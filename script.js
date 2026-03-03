function runAI() {
    const input = document.getElementById("inputText").value;

    fetch("https://your-backend-url/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
    })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").textContent = data.output;
        })
        .catch(err => {
            document.getElementById("result").textContent = "Error running AI.";
            console.error(err);
        });
  }