import { createPredictor } from "./inference.js";

const MODEL_BASE = "./model/v1/";
const WELCOME = "Hi — type a message and I’ll pick the emoji that fits.";

const elements = {
    status: document.getElementById("modelStatus"),
    statusText: document.getElementById("statusText"),
    statusProgress: document.getElementById("statusProgress"),
    chatSubtitle: document.getElementById("chatSubtitle"),
    chatLog: document.getElementById("chatLog"),
    composer: document.getElementById("composer"),
    input: document.getElementById("messageInput"),
    send: document.getElementById("sendBtn"),
    clear: document.getElementById("clearBtn")
};

let predictor = null;
let busy = false;
let conversation = 0;

function setStatus(state, text, progressText = "") {
    elements.status.dataset.state = state;
    elements.statusText.textContent = text;
    elements.statusProgress.textContent = progressText;
    elements.statusProgress.hidden = progressText === "";
}

function scrollChat(behavior = "smooth") {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    elements.chatLog.scrollTo({
        top: elements.chatLog.scrollHeight,
        behavior: reduceMotion ? "auto" : behavior
    });
}

function makeMessage(role) {
    const message = document.createElement("div");
    message.className = `message message-${role}`;
    return message;
}

function showWelcome() {
    const message = makeMessage("bot");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = WELCOME;
    message.append(bubble);
    elements.chatLog.append(message);
}

function addUserMessage(text) {
    const message = makeMessage("user");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    message.append(bubble);
    elements.chatLog.append(message);
}

function addBotError(text) {
    const message = makeMessage("bot");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    message.append(bubble);
    elements.chatLog.append(message);
}

function addPrediction(result) {
    const best = result.predictions[0];
    if (!best) return;

    const message = makeMessage("bot");

    const label = document.createElement("p");
    label.className = "bot-label";
    label.textContent = "EmojiBot";

    const card = document.createElement("div");
    card.className = "prediction";

    const emoji = document.createElement("span");
    emoji.className = "prediction-emoji";
    emoji.textContent = best.emoji;
    emoji.setAttribute("role", "img");
    emoji.setAttribute("aria-label", `Suggested emoji: ${best.emoji}`);

    const copy = document.createElement("div");
    copy.className = "prediction-copy";

    const title = document.createElement("p");
    title.className = "prediction-title";
    const titleText = document.createElement("span");
    titleText.textContent = "Best match";
    const confidence = document.createElement("span");
    confidence.className = "prediction-confidence";
    confidence.textContent = `${Math.round(best.probability * 100)}% confident`;
    title.append(titleText, confidence);

    const meter = document.createElement("div");
    meter.className = "meter";
    meter.setAttribute("role", "meter");
    meter.setAttribute("aria-label", "Prediction confidence");
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", String(Math.round(best.probability * 100)));
    const meterFill = document.createElement("div");
    meterFill.className = "meter-fill";
    meterFill.style.width = `${Math.max(4, best.probability * 100)}%`;
    meter.append(meterFill);

    const alternatives = document.createElement("p");
    alternatives.className = "alternatives";
    alternatives.append(document.createTextNode("Also "));
    result.predictions.slice(1).forEach((prediction) => {
        const chip = document.createElement("span");
        chip.className = "alt-chip";
        chip.textContent = `${prediction.emoji} ${Math.round(prediction.probability * 100)}%`;
        chip.title = "Alternative prediction";
        alternatives.append(chip);
    });

    copy.append(title, meter, alternatives);
    card.append(emoji, copy);
    message.append(label, card);
    elements.chatLog.append(message);
    scrollChat();
}

function showTyping() {
    const message = makeMessage("bot");
    message.setAttribute("aria-label", "EmojiBot is thinking");

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const typing = document.createElement("span");
    typing.className = "typing";
    typing.setAttribute("aria-hidden", "true");
    typing.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));

    bubble.append(typing);
    message.append(bubble);
    elements.chatLog.append(message);
    scrollChat();
    return message;
}

function setComposerBusy(value) {
    busy = value;
    elements.send.disabled = value || !predictor;
    elements.input.disabled = value || !predictor;
    elements.clear.disabled = !predictor;
}

function readWithProgress(response, total, onProgress) {
    if (!response.body || !response.body.getReader) {
        return response.arrayBuffer();
    }

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    const pump = () =>
        reader.read().then(({ done, value }) => {
            if (done) {
                const output = new Uint8Array(received);
                let offset = 0;
                for (const chunk of chunks) {
                    output.set(chunk, offset);
                    offset += chunk.byteLength;
                }
                return output.buffer;
            }

            chunks.push(value);
            received += value.byteLength;
            onProgress(received, total);
            return pump();
        });

    return pump();
}

async function loadModel() {
    setStatus("loading", "Loading the model…", "0%");

    try {
        const [meta, vocab, labels] = await Promise.all([
            fetch(`${MODEL_BASE}model.json`).then((response) => {
                if (!response.ok) throw new Error(`model.json: ${response.status}`);
                return response.json();
            }),
            fetch(`${MODEL_BASE}vocab.json`).then((response) => {
                if (!response.ok) throw new Error(`vocab.json: ${response.status}`);
                return response.json();
            }),
            fetch(`${MODEL_BASE}emojis.json`).then((response) => {
                if (!response.ok) throw new Error(`emojis.json: ${response.status}`);
                return response.json();
            })
        ]);

        const response = await fetch(`${MODEL_BASE}model.bin`);
        if (!response.ok) throw new Error(`model.bin: ${response.status}`);

        const contentLength = Number(response.headers.get("content-length")) || 0;
        const total = contentLength || meta.sizeBytes;
        let lastPercent = -1;

        const buffer = await readWithProgress(response, total, (received, expected) => {
            const percent = Math.min(99, Math.round((received / expected) * 100));
            if (percent === lastPercent) return;
            lastPercent = percent;
            setStatus("loading", "Loading the model…", `${percent}%`);
        });

        predictor = createPredictor({ buffer, meta, vocab, labels });

        const size = Math.round(meta.sizeBytes / (1024 * 1024));
        setStatus("ready", "Model ready · runs locally", `${size} MB`);
        elements.chatSubtitle.textContent = "On-device · ready";
        setComposerBusy(false);
        elements.input.focus();
    } catch (error) {
        console.error(error);
        predictor = null;
        setStatus("error", "Model could not load", "");
        elements.chatSubtitle.textContent = "Unavailable";
        addBotError("The model could not be downloaded. Please refresh the page to try again.");
        scrollChat("auto");
    }
}

async function sendMessage(text) {
    const value = text.trim();
    if (!value || !predictor || busy) return;

    const currentConversation = conversation;
    addUserMessage(value);
    elements.input.value = "";
    setComposerBusy(true);

    const typing = showTyping();

    try {
        await new Promise((resolve) => requestAnimationFrame(() => resolve()));
        const result = predictor.predict(value);
        if (currentConversation !== conversation) return;
        typing.remove();
        addPrediction(result);
    } catch (error) {
        console.error(error);
        if (currentConversation !== conversation) return;
        typing.remove();
        addBotError("Something went wrong while making a prediction. Try a shorter message.");
    } finally {
        if (currentConversation === conversation) {
            setComposerBusy(false);
            elements.input.focus();
        }
    }
}

elements.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(elements.input.value);
});

elements.clear.addEventListener("click", () => {
    conversation += 1;
    elements.chatLog.replaceChildren();
    showWelcome();
    setComposerBusy(false);
    elements.input.focus();
});

document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        const text = chip.dataset.example || chip.textContent;
        if (!predictor) {
            elements.input.value = text;
            elements.input.focus();
            return;
        }
        elements.input.value = text;
        sendMessage(text);
    });
});

showWelcome();
setComposerBusy(true);
loadModel();
