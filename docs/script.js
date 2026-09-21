const form = document.querySelector("#review-form");
const output = document.querySelector("#review-output");

const fields = {
  summary: document.querySelector("#paper-summary"),
  figure: document.querySelector("#figure-context"),
  novelty: document.querySelector("#novelty-context"),
  quality: document.querySelector("#quality-weight"),
  figureWeight: document.querySelector("#figure-weight"),
  noveltyWeight: document.querySelector("#novelty-weight"),
};

const scores = {
  quality: document.querySelector("#score-quality"),
  figure: document.querySelector("#score-figure"),
  novelty: document.querySelector("#score-novelty"),
};

function firstSentence(text) {
  const cleaned = text.trim().replace(/\s+/g, " ");
  const match = cleaned.match(/.*?[.!?](\s|$)/);
  return match ? match[0].trim() : cleaned;
}

function score(value) {
  return (Number(value) / 100).toFixed(2);
}

function renderReview() {
  const quality = Number(fields.quality.value);
  const figure = Number(fields.figureWeight.value);
  const novelty = Number(fields.noveltyWeight.value);
  const summary = firstSentence(fields.summary.value) || "The manuscript presents a research contribution.";
  const figureContext = firstSentence(fields.figure.value) || "The figure context is not provided.";
  const noveltyContext = firstSentence(fields.novelty.value) || "The novelty context is not provided.";

  scores.quality.textContent = score(quality);
  scores.figure.textContent = score(figure);
  scores.novelty.textContent = score(novelty);

  const critique =
    quality > 65
      ? "The review should balance summary, strengths, limitations, and actionable suggestions."
      : "With less quality emphasis, the review may become thinner and less useful to authors.";
  const figureLine =
    figure > 50
      ? `The figure-grounded feedback should explicitly address this visual evidence: ${figureContext}`
      : "The review gives less priority to visual evidence, so figure-specific comments may be sparse.";
  const noveltyLine =
    novelty > 50
      ? `The novelty assessment should be calibrated against the outside literature signal: ${noveltyContext}`
      : "The review gives less priority to external novelty grounding, which can weaken claims about originality.";

  output.innerHTML = `
    <p><strong>Summary.</strong> ${summary}</p>
    <p><strong>Strengths.</strong> The work appears to target an important scientific-review bottleneck and provides enough structure for a reviewer to evaluate the core claim.</p>
    <p><strong>Contextual critique.</strong> ${figureLine}</p>
    <p><strong>Novelty critique.</strong> ${noveltyLine}</p>
    <p><strong>Actionable suggestion.</strong> ${critique} A stronger final review would connect each major criticism to manuscript evidence and to the auxiliary context used during evaluation.</p>
  `;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  renderReview();
});

Object.values(fields).forEach((field) => {
  field.addEventListener("input", renderReview);
});

renderReview();

const copyButton = document.querySelector("#copy-bibtex");
const bibtexText = document.querySelector("#bibtex-text");

if (copyButton && bibtexText) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtexText.textContent);
      copyButton.textContent = "Copied!";
    } catch (err) {
      copyButton.textContent = "Copy failed";
    }
    setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1800);
  });
}
