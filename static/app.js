let currentDocumentId = null;

const pdfInput = document.querySelector("#pdf");
const uploadBtn = document.querySelector("#uploadBtn");
const askBtn = document.querySelector("#askBtn");
const uploadStatus = document.querySelector("#uploadStatus");
const questionInput = document.querySelector("#question");
const answerBox = document.querySelector("#answer");
const sourcesBox = document.querySelector("#sources");

function setBusy(button, isBusy) {
  button.disabled = isBusy;
}

async function readError(response, fallback) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const error = await response.json();
    return error.detail || fallback;
  }

  const text = await response.text();
  return text || fallback;
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a PDF first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setBusy(uploadBtn, true);
  uploadStatus.textContent = "Indexing document...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await readError(response, "Upload failed."));
    }

    const result = await response.json();
    currentDocumentId = result.document_id;
    uploadStatus.textContent = `${result.filename} indexed with ${result.chunks_indexed} chunks.`;
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    setBusy(uploadBtn, false);
  }
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) {
    answerBox.textContent = "Ask a question first.";
    return;
  }
  if (!currentDocumentId) {
    answerBox.textContent = "Upload and index a PDF before asking a question.";
    return;
  }

  setBusy(askBtn, true);
  answerBox.textContent = "Retrieving context and generating an answer...";
  sourcesBox.innerHTML = "";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        document_id: currentDocumentId,
      }),
    });

    if (!response.ok) {
      throw new Error(await readError(response, "Question failed."));
    }

    const result = await response.json();
    answerBox.textContent = result.answer;
    sourcesBox.innerHTML = result.sources
      .map(
        (source) => `
          <article class="source">
            <strong>${source.filename} ${source.page ? `- page ${source.page}` : ""}</strong>
            <p>${source.preview}</p>
          </article>
        `
      )
      .join("");
  } catch (error) {
    answerBox.textContent = error.message;
  } finally {
    setBusy(askBtn, false);
  }
});
