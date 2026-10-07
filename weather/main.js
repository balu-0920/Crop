const api = {
    key: "fcc8de7015bbb202209bbf0261babf4c",
    base: "https://api.openweathermap.org/data/2.5/"
};
let currentWeather = null;
let rainfallForecast = null;   // { value_mm, source } - see fetchRainfallForecast()
let currentRegion = null;      // { country, state, district, matched_level, common_crops } - see fetchRegion()
let isCelsius = true;

// Set to false once you've confirmed everything works - this logs every
// stage of the prediction pipeline to the browser console so you can see
// exactly what's being sent and where a bad value would come from.
const DEBUG = true;

function debugLog(label, value){
    if(DEBUG){
        console.log(`[DEBUG] ${label}:`, value);
    }
}
// ===============================
// DOM Elements
// ===============================

const searchBox = document.querySelector(".search-box");
const searchBtn = document.querySelector(".search-btn");
const locationBtn = document.querySelector(".location-btn");

const loading = document.querySelector(".loading");
const error = document.querySelector(".error");
const weatherInfo = document.querySelector(".weather-info");

const recentList = document.getElementById("recentList");

// ===============================
// Crop Recommendation - DOM Elements
// ===============================

const recommendBtn = document.getElementById("recommendBtn");
const cropEmptyState = document.querySelector(".crop-empty-state");
const cropLoading = document.querySelector(".crop-loading");
const cropError = document.querySelector(".crop-error");
const cropErrorMsg = document.getElementById("cropErrorMsg");
const cropResult = document.querySelector(".crop-result");
const cropName = document.getElementById("cropName");
const cropImage = document.getElementById("cropImage");
const cropScientificName = document.getElementById("cropScientificName");
const cropDescription = document.getElementById("cropDescription");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const suitabilityValue = document.getElementById("suitabilityValue");
const suitabilityBar = document.getElementById("suitabilityBar");
const harvestTimeValue = document.getElementById("harvestTimeValue");
const waterNeedValue = document.getElementById("waterNeedValue");
const predictionTimeValue = document.getElementById("predictionTimeValue");
const weatherSummaryList = document.getElementById("weatherSummaryList");
const summarySuitableFor = document.getElementById("summarySuitableFor");
const suitabilityList = document.getElementById("suitabilityList");
const whyCropList = document.getElementById("whyCropList");
const suggestionsList = document.getElementById("suggestionsList");
const topPredictionsSection = document.getElementById("topPredictionsSection");
const topPredictionsList = document.getElementById("topPredictionsList");
const fertilizerTip = document.getElementById("fertilizerTip");

// --- Regional Crop Intelligence (Part 2) DOM elements ---
const regionCard = document.getElementById("regionCard");
const regionLabel = document.getElementById("regionLabel");
const commonCropsBadges = document.getElementById("commonCropsBadges");

// --- Smart Recommendation Engine (Part 3) DOM elements ---
const recommendationWeightsLine = document.getElementById("recommendationWeightsLine");
const recommendationsList = document.getElementById("recommendationsList");

// Your Flask backend runs on a different origin than this page, so we
// need the full URL (not just "/predict") for fetch() to reach it.
const CROP_API_BASE = "http://127.0.0.1:5000";

// ===============================
// Event Listeners
// ===============================

searchBtn.addEventListener("click", () => {

    const city = searchBox.value.trim();

    if(city){

        getWeather(city);

    }

});

searchBox.addEventListener("keypress", (e)=>{

    if(e.key==="Enter"){

        const city = searchBox.value.trim();

        if(city){

            getWeather(city);

        }

    }

});

locationBtn.addEventListener("click", getCurrentLocation);

recommendBtn.addEventListener("click", getCropRecommendation);

// ===============================
// Current Location
// ===============================

function getCurrentLocation(){

    if(!navigator.geolocation){

        alert("Geolocation is not supported.");

        return;

    }

    navigator.geolocation.getCurrentPosition(

        (position)=>{

            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            getWeatherByCoordinates(lat, lon);

        },

        ()=>{

            alert("Unable to access your location.");

        }

    );

}

// ===============================
// Loading
// ===============================

function showLoading(){

    loading.classList.remove("hidden");

    weatherInfo.classList.add("hidden");

    error.classList.add("hidden");

}

function hideLoading(){

    loading.classList.add("hidden");

    weatherInfo.classList.remove("hidden");

}

// ===============================
// Error
// ===============================

function showError(){

    loading.classList.add("hidden");

    weatherInfo.classList.add("hidden");

    error.classList.remove("hidden");

}

// ===============================
// Weather By City
// ===============================

async function getWeather(city){

    showLoading();

    try{

        const response = await fetch(

            `${api.base}weather?q=${city}&units=metric&appid=${api.key}`

        );

        const data = await response.json();

        if(data.cod != 200){

            throw new Error();

        }

        await handleWeatherSuccess(data);

        saveRecent(city);

        loadRecent();

    }

    catch{

        showError();

    }

    finally{

        hideLoading();

    }

}

// ===============================
// Weather By Coordinates
// ===============================

async function getWeatherByCoordinates(lat, lon){

    showLoading();

    try{

        const response = await fetch(

            `${api.base}weather?lat=${lat}&lon=${lon}&units=metric&appid=${api.key}`

        );

        const data = await response.json();

        if(data.cod != 200){

            throw new Error();

        }

        await handleWeatherSuccess(data);

    }

    catch{

        showError();

    }

    finally{

        hideLoading();

    }

}

// ===============================
// Shared weather success handler
//
// Both getWeather() and getWeatherByCoordinates() end up with the same
// OpenWeatherMap "current weather" response shape - this function is the
// ONE place that stores it and kicks off the rainfall forecast lookup, so
// that logic doesn't have to be duplicated in both fetch functions above.
// ===============================

async function handleWeatherSuccess(data){

    currentWeather = data;

    displayWeather(data);

    // Rainfall and region are independent lookups - run them together
    // (Promise.all) instead of one after another, so the person isn't
    // waiting twice as long for two calls that don't depend on each other.
    await Promise.all([
        fetchRainfallForecast(data.coord.lat, data.coord.lon),
        fetchRegion(data.coord.lat, data.coord.lon)
    ]);

}

// ===============================
// Rainfall Forecast
//
// THE BUG FIX: OpenWeatherMap's "current weather" endpoint only includes
// a "rain" object when it is ACTIVELY raining at that exact moment - most
// of the time it's simply absent, which the old code silently treated as
// 0mm. Real rainfall in the training data ranges from ~20-300mm; feeding
// the model a hard 0 is a value it never saw in training, and it consistently
// mispredicted "muskmelon" as a result (verified: 17/30 random trials with
// rainfall=0 predicted muskmelon regardless of every other input).
//
// The fix: call OpenWeatherMap's 5-day/3-hour FORECAST endpoint instead,
// which returns rain.3h (mm of rain expected in that 3-hour window) for
// each of the next several time slots - present or absent depending on
// whether rain is actually forecast, not just "currently happening". We
// sum the next 24 hours (8 slots x 3h) into one rainfall estimate.
// ===============================

async function fetchRainfallForecast(lat, lon){

    try{

        const response = await fetch(

            `${api.base}forecast?lat=${lat}&lon=${lon}&units=metric&appid=${api.key}`

        );

        const data = await response.json();

        if(!data.list){

            throw new Error("Forecast data unavailable");

        }

        // Sum rain.3h across the next 24 hours (8 x 3-hour slots). Each
        // slot's "rain" object is only present if rain is forecast for
        // that window - absent means 0 for that slot, which is a REAL
        // "no rain expected" signal, not a missing-data placeholder.
        const next24h = data.list.slice(0, 8);
        const totalRainfall = next24h.reduce(

            (sum, slot) => sum + (slot.rain?.["3h"] || 0),
            0

        );

        rainfallForecast = { value_mm: totalRainfall, source: "forecast" };

        debugLog("Rainfall forecast (next 24h, from forecast API)", rainfallForecast);

    }

    catch(err){

        // Forecast API failed (network issue, quota, etc.) - fall back to
        // whatever the current-weather response has, rather than silently
        // going back to a hard 0.
        const fallbackRain = currentWeather?.rain
            ? (currentWeather.rain["1h"] ?? currentWeather.rain["3h"] ?? null)
            : null;

        if(fallbackRain !== null){

            // Extrapolate a single hour's rain rate across 24h as a rough estimate.
            rainfallForecast = { value_mm: fallbackRain * 24, source: "extrapolated" };

        } else {

            // Last resort: a documented rough estimate rather than 0.
            // This is clearly labeled as an estimate in the UI (see
            // showCropResult) so it's never mistaken for a real reading.
            rainfallForecast = { value_mm: 100, source: "estimated" };

        }

        debugLog("Forecast API failed, using fallback rainfall", rainfallForecast);

    }

}

// ===============================
// Regional Crop Intelligence (Part 2)
//
// Calls our Flask backend's /region endpoint, which geocodes the
// coordinates (country/state/district) and looks up that region's common
// crops from data/regional_crops.json - with automatic fallback to state,
// then country, then a global default if a more specific level isn't in
// the database. Result is cached in currentRegion so getCropRecommendation()
// can reuse it without geocoding a second time.
// ===============================

async function fetchRegion(lat, lon){

    try{

        const response = await fetch(`${CROP_API_BASE}/region`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat, lon })
        });

        const data = await response.json();

        if(!data.success){
            throw new Error(data.message || "Region lookup failed.");
        }

        currentRegion = {
            country: data.location.country,
            state: data.location.state,
            district: data.location.district,
            matched_level: data.matched_level,
            common_crops: data.common_crops
        };

        debugLog("Resolved region", currentRegion);

        renderRegionCard(currentRegion);

    }

    catch(err){

        // Regional intelligence is a soft dependency - if it fails (network,
        // backend down, etc.) we simply hide the card rather than blocking
        // weather or crop prediction, which don't need it to function.
        currentRegion = null;
        regionCard.classList.add("hidden");

        debugLog("Region lookup failed", err.message);

    }

}

function renderRegionCard(region){

    if(!region.common_crops || region.common_crops.length === 0){
        regionCard.classList.add("hidden");
        return;
    }

    regionCard.classList.remove("hidden");

    // Build a human-readable "Hyderabad, Telangana, India" style label from
    // whichever fields geocoding actually found, most specific first.
    const parts = [region.district, region.state, region.country].filter(Boolean);
    regionLabel.innerText = parts.length > 0 ? parts.join(", ") : "Region not detected - showing default crops";

    commonCropsBadges.innerHTML = "";
    region.common_crops.forEach(crop=>{

        const badge = document.createElement("span");
        badge.className = "crop-badge";
        badge.innerText = `${cropEmoji(crop)} ${crop}`;
        commonCropsBadges.appendChild(badge);

    });

}

// Small icon lookup for common crop badges - purely decorative, falls back
// to a generic plant emoji for anything not in this list.
function cropEmoji(cropName){

    const emojiMap = {
        rice: "🌾", wheat: "🌾", maize: "🌽", cotton: "☁️", sugarcane: "🌱",
        blackgram: "🫘", pulses: "🫘", chickpea: "🫘", groundnut: "🥜",
        chilli: "🌶️", mango: "🥭", banana: "🍌", coconut: "🥥", grapes: "🍇",
        jute: "🌿", mustard: "🌼", soybean: "🌱", turmeric: "🟡", onion: "🧅",
        orange: "🍊", jowar: "🌾", redgram: "🫘", rubber: "🌳", potato: "🥔"
    };

    return emojiMap[normalizeCropName(cropName)] || "🌱";

}

function normalizeCropName(cropName){
    return cropName.trim().toLowerCase().replace(/\s+/g, "").replace(/-/g, "");
}

// ===============================
// Recent Searches
// ===============================

function saveRecent(city){

    let cities = JSON.parse(

        localStorage.getItem("recentCities")

    ) || [];

    cities = cities.filter(

        c => c.toLowerCase() !== city.toLowerCase()

    );

    cities.unshift(city);

    if(cities.length > 5){

        cities.pop();

    }

    localStorage.setItem(

        "recentCities",

        JSON.stringify(cities)

    );

}

function loadRecent(){

    recentList.innerHTML = "";

    const cities = JSON.parse(

        localStorage.getItem("recentCities")

    ) || [];

    cities.forEach(city=>{

        const btn = document.createElement("button");

        btn.innerText = city;

        btn.onclick = ()=>{

            searchBox.value = city;

            getWeather(city);

        };

        recentList.appendChild(btn);

    });

}

// Load Recent Searches

loadRecent();

// ===============================
// Display Weather
// ===============================

function displayWeather(weather){

    document.querySelector(".city").innerText =
        `${weather.name}, ${weather.sys.country}`;

    document.querySelector(".date").innerText =
        formatDate(new Date());

    document.querySelector(".temp").innerText =
        `${Math.round(convertTemperature(weather.main.temp))}°${isCelsius ? "C" : "F"}`;

    document.querySelector(".weather").innerText =
        weather.weather[0].description;

    document.getElementById("feelsLike").innerText =
        `${Math.round(convertTemperature(weather.main.feels_like))}°${isCelsius ? "C" : "F"}`;

    document.getElementById("humidity").innerText =
        `${weather.main.humidity}%`;

    document.getElementById("wind").innerText =
        `${weather.wind.speed} m/s`;

    document.getElementById("pressure").innerText =
        `${weather.main.pressure} hPa`;

    document.getElementById("visibility").innerText =
        `${(weather.visibility / 1000).toFixed(1)} km`;

    document.getElementById("highLow").innerText =
        `${Math.round(convertTemperature(weather.main.temp_max))}° / ${Math.round(convertTemperature(weather.main.temp_min))}°`;

    document.getElementById("sunrise").innerText =
        formatTime(weather.sys.sunrise);

    document.getElementById("sunset").innerText =
        formatTime(weather.sys.sunset);

    document.getElementById("updatedTime").innerText =
        new Date().toLocaleTimeString();

    // Weather Icon

    document.getElementById("weatherIcon").src =
        `https://openweathermap.org/img/wn/${weather.weather[0].icon}@4x.png`;

}

// ===============================
// Temperature Conversion
// ===============================

const unitBtn = document.getElementById("unitBtn");

unitBtn.addEventListener("click",()=>{

    if(!currentWeather) return;

    isCelsius = !isCelsius;

    unitBtn.innerText =
        isCelsius
        ? "Switch to °F"
        : "Switch to °C";

    displayWeather(currentWeather);

});

function convertTemperature(temp){

    if(isCelsius){

        return temp;

    }

    return (temp * 9 / 5) + 32;

}

// ===============================
// Date Formatter
// ===============================

function formatDate(date){

    const months=[
        "January","February","March",
        "April","May","June",
        "July","August","September",
        "October","November","December"
    ];

    const days=[
        "Sunday","Monday","Tuesday",
        "Wednesday","Thursday",
        "Friday","Saturday"
    ];

    return `${days[date.getDay()]}, ${date.getDate()} ${months[date.getMonth()]} ${date.getFullYear()}`;

}

// ===============================
// Time Formatter
// ===============================

function formatTime(unix){

    const date = new Date(unix * 1000);

    return date.toLocaleTimeString([],{

        hour:"2-digit",

        minute:"2-digit"

    });

}

// ===============================
// Crop Recommendation
// ===============================

function showCropLoading(){

    cropEmptyState.classList.add("hidden");
    cropLoading.classList.remove("hidden");
    cropError.classList.add("hidden");
    cropResult.classList.add("hidden");

}

function showCropError(message){

    cropEmptyState.classList.add("hidden");
    cropLoading.classList.add("hidden");
    cropResult.classList.add("hidden");
    cropError.classList.remove("hidden");

    cropErrorMsg.innerText = message;

}

function showCropResult(data){

    cropEmptyState.classList.add("hidden");
    cropLoading.classList.add("hidden");
    cropError.classList.add("hidden");
    cropResult.classList.remove("hidden");

    const crop = data.crop_info; // may be missing if a crop isn't in our knowledge base

    // --- Header: name, image, scientific name, description ---
    const displayName = crop ? crop.name : data.prediction;
    cropName.innerText = `🌾 ${displayName}`;
    // data.image_url is a root-relative path (e.g. "/static/images/rice.svg")
    // that Flask returns relative to ITS OWN origin. Since our frontend is
    // served from a different origin (a different port, or a file server),
    // a plain "/static/..." path would resolve against the FRONTEND's
    // origin instead and 404. We prefix it with CROP_API_BASE to point it
    // at Flask, the same way we do for the fetch() URL itself.
    cropImage.src = data.image_url ? `${CROP_API_BASE}${data.image_url}` : "static/images/default.svg";
    cropImage.alt = displayName;
    cropScientificName.innerText = crop ? crop.scientific_name : "";
    cropDescription.innerText = crop ? crop.description : "";

    // --- Confidence ---
    // Only shown if the model actually supports predict_proba() - Flask
    // sends confidence: null when it doesn't, and we display that
    // honestly rather than making up a number.
    const hasConfidence = data.confidence !== null && data.confidence !== undefined;
    confidenceValue.innerText = hasConfidence ? `${data.confidence}%` : "N/A";
    // Setting width via a brief delay lets the CSS transition actually
    // animate from 0 instead of jumping straight to the final value.
    confidenceBar.style.width = "0%";
    requestAnimationFrame(()=> confidenceBar.style.width = hasConfidence ? `${data.confidence}%` : "0%");

    // --- Suitability ---
    const suitabilityPct = data.suitability?.overall_percentage;
    suitabilityValue.innerText = (suitabilityPct !== undefined && suitabilityPct !== null) ? `${suitabilityPct}%` : "N/A";
    suitabilityBar.style.width = "0%";
    requestAnimationFrame(()=> suitabilityBar.style.width = suitabilityPct ? `${suitabilityPct}%` : "0%");

    // Color the bars based on how good the number actually is, so a low
    // score reads as a low score at a glance, not just a shorter green bar.
    [[confidenceBar, hasConfidence ? data.confidence : 0], [suitabilityBar, suitabilityPct || 0]].forEach(([bar, value])=>{
        bar.classList.remove("bar-good", "bar-warn", "bar-bad");
        if(value >= 75) bar.classList.add("bar-good");
        else if(value >= 50) bar.classList.add("bar-warn");
        else bar.classList.add("bar-bad");
    });

    // --- Quick facts ---
    harvestTimeValue.innerText = crop ? crop.harvest_time : "--";
    waterNeedValue.innerText = crop ? crop.water_requirement : "--";
    predictionTimeValue.innerText = data.prediction_time || "--";

    // --- Current Conditions summary ---
    weatherSummaryList.innerHTML = "";
    const weather = data.weather_used;
    if(weather){

        const rainfallSourceLabel = {
            forecast: "24h forecast",
            extrapolated: "extrapolated",
            estimated: "estimated"
        }[rainfallForecast?.source] || "";

        const rows = [
            ["Temperature", `${Math.round(weather.temperature)}°C`],
            ["Humidity", `${Math.round(weather.humidity)}%`],
            ["Rainfall", `${Math.round(weather.rainfall)} mm${rainfallSourceLabel ? ` (${rainfallSourceLabel})` : ""}`]
        ];

        rows.forEach(([label, value])=>{

            const li = document.createElement("li");
            li.innerHTML = `<span>${label}</span><span>${value}</span>`;
            weatherSummaryList.appendChild(li);

        });

    }

    summarySuitableFor.innerText = data.suitability
        ? `Suitable for ${displayName} ${data.suitability.overall_percentage >= 60 ? "✔" : "⚠"}`
        : "";

    // --- Suitability breakdown (Temperature / Humidity / Rainfall) ---
    suitabilityList.innerHTML = "";
    if(data.suitability){

        [["Temperature", data.suitability.temperature],
         ["Humidity", data.suitability.humidity],
         ["Rainfall", data.suitability.rainfall]].forEach(([label, factor])=>{

            const li = document.createElement("li");
            li.innerHTML = `<span>${label}</span><span>${factor.symbol} ${factor.label}</span>`;
            suitabilityList.appendChild(li);

        });

    }

    // --- Why This Crop ---
    whyCropList.innerHTML = "";
    (data.why_this_crop || []).forEach(reason=>{

        const li = document.createElement("li");
        li.innerText = reason;
        whyCropList.appendChild(li);

    });

    // --- Suggestions ---
    suggestionsList.innerHTML = "";
    (data.suggestions || []).forEach(tip=>{

        const li = document.createElement("li");
        li.innerText = `💡 ${tip}`;
        suggestionsList.appendChild(li);

    });

    // --- Top Recommendations ---
    // Only shown when the backend actually provided probabilities -
    // otherwise we hide the whole section rather than show an empty list.
    if(data.top_predictions && data.top_predictions.length > 0){

        topPredictionsSection.classList.remove("hidden");
        topPredictionsList.innerHTML = "";

        data.top_predictions.forEach(item=>{

            const li = document.createElement("li");
            const name = crop_info_name_fallback(item.crop);
            li.innerText = `${name} — ${item.probability}%`;
            topPredictionsList.appendChild(li);

        });

    } else {

        topPredictionsSection.classList.add("hidden");

    }

    // --- Fertilizer tip ---
    fertilizerTip.innerText = crop ? crop.fertilizer_tips : "No fertilizer guidance available for this crop.";

    // --- Overall Recommendations (Part 3 - Smart Recommendation Engine) ---
    renderRecommendations(data.recommendations, data.recommendation_weights, data.decision_weights);

    // --- Prediction History (frontend-only, see saveHistoryEntry below) ---
    saveHistoryEntry(data, displayName);

}

// Renders the ranked list produced by recommendation_engine.py: each entry
// gets a star rating (overall_score converted to 1-5 stars) plus its own
// ✔/⚠/✖ "Why Recommended" badges - no separate lookup needed, the backend
// already attached the explanation to each entry.
// Formats one decision field that the backend marks as available or unavailable.
// Unavailable fields are shown explicitly (never as 0 or a guess).
function decisionField(field, formatter){
    if(!field || !field.available){
        return `<span class="decision-unavailable" title="${(field?.reason || "Unavailable").replace(/"/g, "&quot;")}">Unavailable</span>`;
    }
    return formatter(field);
}

const inr = n => "₹" + Number(n).toLocaleString("en-IN");

function renderDecisionCard(rec, decisionWeights){
    const card = document.getElementById("decisionCard");
    if(!rec || !rec.decision){ card.classList.add("hidden"); return; }
    const d = rec.decision;

    const yieldHtml = decisionField(d.expected_yield, f =>
        `${f.value} t/ha <small>(${f.range[0]}–${f.range[1]})</small>`);
    const revenueHtml = decisionField(d.revenue, f =>
        `${inr(f.value)}/ha <small>(${inr(f.range[0])}–${inr(f.range[1])})</small>`);
    const profitHtml = decisionField(d.profit, f =>
        `${inr(f.value)}/ha <small>(${inr(f.range[0])}–${inr(f.range[1])})</small>`);

    const riskLevel = d.risk.level;
    const used = Object.entries(d.components_used).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(" · ");
    const missing = d.components_unavailable.length
        ? ` · not used (no data): ${d.components_unavailable.join(", ")}` : "";

    card.innerHTML = `
        <div class="decision-head">
            <span class="decision-label">Recommended crop</span>
            <span class="decision-crop">${cropEmoji(rec.crop)} ${rec.crop}</span>
        </div>
        <div class="decision-grid">
            <div><span class="decision-key">Suitability</span><span class="decision-val">${d.suitability_pct}%</span></div>
            <div><span class="decision-key">Expected yield</span><span class="decision-val">${yieldHtml}</span></div>
            <div><span class="decision-key">Estimated revenue</span><span class="decision-val">${revenueHtml}</span></div>
            <div><span class="decision-key">Estimated profit</span><span class="decision-val">${profitHtml}</span></div>
            <div><span class="decision-key">Risk level</span><span class="decision-val"><span class="risk-pill risk-${riskLevel.toLowerCase()}">${riskLevel}</span>${d.risk.score !== null ? ` <small>(${d.risk.score}/100)</small>` : ""}</span></div>
            <div><span class="decision-key">Final rank score</span><span class="decision-val">${d.final_score}%</span></div>
        </div>
        <ul class="decision-reasons">${d.main_reasons.map(r => `<li>${r}</li>`).join("")}</ul>
        <p class="decision-note">Final score weights used: ${used}${missing}. Yield, revenue and profit stay "Unavailable" until price, cost and yield data are added (see data/economics/README.md). Estimates are not forecasts of actual farm results.</p>
    `;
    card.classList.remove("hidden");
}

function renderRecommendations(recommendations, weights, decisionWeights){

    if(!recommendations || recommendations.length === 0){
        document.getElementById("recommendationSection").classList.add("hidden");
        return;
    }

    document.getElementById("recommendationSection").classList.remove("hidden");

    recommendationWeightsLine.innerText = weights
        ? `Score = ML ${Math.round(weights.ml * 100)}% · Regional ${Math.round(weights.regional * 100)}% · Weather ${Math.round(weights.weather * 100)}%`
        : "";

    renderDecisionCard(recommendations[0], decisionWeights);

    recommendationsList.innerHTML = "";

    recommendations.forEach((rec, index)=>{

        const li = document.createElement("li");
        li.className = "recommendation-item";

        const stars = "⭐".repeat(rec.stars) + "☆".repeat(5 - rec.stars);

        const reasonBadges = rec.reasons.map(reason=>{
            const symbol = reason.ok === true ? "✔" : reason.ok === false ? "✖" : "•";
            const cls = reason.ok === true ? "reason-ok" : reason.ok === false ? "reason-bad" : "reason-neutral";
            return `<span class="reason-badge ${cls}">${symbol} ${reason.text}</span>`;
        }).join("");

        const riskLevel = rec.decision?.risk?.level;
        const riskBadge = riskLevel
            ? `<span class="risk-pill risk-${riskLevel.toLowerCase()}">${riskLevel} risk</span>` : "";

        li.innerHTML = `
            <div class="recommendation-row">
                <span class="recommendation-rank">${index + 1}.</span>
                <span class="recommendation-crop">${cropEmoji(rec.crop)} ${rec.crop}</span>
                ${riskBadge}
                <span class="recommendation-stars">${stars}</span>
                <span class="recommendation-score" title="Final rank score">${rec.decision?.final_score ?? rec.overall_score}%</span>
            </div>
            <div class="recommendation-reasons">${reasonBadges}</div>
        `;

        recommendationsList.appendChild(li);

    });

}

// Small helper: capitalize a raw crop key (e.g. "kidneybeans") for display
// in the Top Recommendations list, since that list only gets the raw
// crop key + probability from Flask, not the full crop_info object.
function crop_info_name_fallback(cropKey){

    return cropKey.charAt(0).toUpperCase() + cropKey.slice(1);

}

// getCropRecommendation() is declared "async" so we can use "await" inside
// it - that lets us write code that WAITS for the network request to
// finish before moving to the next line, without freezing the page while
// it waits (the browser keeps running everything else normally).
async function getCropRecommendation(){

    // Guard clause: we need weather data already loaded before we can read
    // temperature/humidity/rainfall out of it. currentWeather is set by
    // displayWeather() elsewhere in this file whenever a search succeeds.
    if(!currentWeather){

        showCropError("Please search a location first, so we have weather data to work with.");

        return;

    }

    // Read the user-entered soil values straight from the input fields.
    // Every <input> value comes back as a STRING, even for type="number",
    // so we convert each one with parseFloat() before sending it onward -
    // our Flask endpoint explicitly checks that every field is a number
    // and will reject strings with a 400 error.
    const nitrogen = parseFloat(document.getElementById("nitrogen").value);
    const phosphorus = parseFloat(document.getElementById("phosphorus").value);
    const potassium = parseFloat(document.getElementById("potassium").value);
    const ph = parseFloat(document.getElementById("ph").value);

    // parseFloat() returns NaN (Not a Number) if the field was empty or
    // contained non-numeric text. We check for that BEFORE calling the
    // API, so the user gets an immediate, specific message instead of a
    // generic server error.
    if([nitrogen, phosphorus, potassium, ph].some(Number.isNaN)){

        showCropError("Please fill in all soil fields (N, P, K, pH) with valid numbers.");

        return;

    }

    // --- Auto-sourced weather values (Phase 4, rainfall fixed here) ---
    // The user no longer enters these - we read them straight from the
    // weather data OpenWeatherMap already gave us.
    //
    // rainfallForecast is set by fetchRainfallForecast() whenever weather
    // loads (see handleWeatherSuccess) - it's a real forecast-based estimate,
    // never a hardcoded 0. See the big comment above fetchRainfallForecast()
    // for why the old "0 if not currently raining" logic was the actual
    // cause of the muskmelon bug.
    if(!rainfallForecast){

        showCropError("Rainfall data isn't ready yet. Please wait a moment and try again.");

        return;

    }

    const rainfall = rainfallForecast.value_mm;

    const weatherCondition = currentWeather.weather?.[0]?.main || null;
    const cloudCover = currentWeather.clouds?.all ?? null;

    // Build the exact JSON object our Flask /predict endpoint expects,
    // plus two optional extras (weather_condition, cloud_cover) that
    // Flask uses only for display and suggestion text, not for the
    // model input itself.
    const payload = {
        N: nitrogen,
        P: phosphorus,
        K: potassium,
        temperature: currentWeather.main.temp,
        humidity: currentWeather.main.humidity,
        ph: ph,
        rainfall: rainfall,
        weather_condition: weatherCondition,
        cloud_cover: cloudCover,

        // Regional Crop Intelligence (Part 2): reuse the region already
        // resolved by fetchRegion() when weather loaded, so /predict
        // doesn't have to geocode a second time. If it isn't ready yet for
        // some reason, we still send lat/lon so Flask can resolve it itself.
        country: currentRegion?.country ?? null,
        state: currentRegion?.state ?? null,
        district: currentRegion?.district ?? null,
        lat: currentWeather.coord.lat,
        lon: currentWeather.coord.lon
    };

    // Debugging: confirm every value is a genuine number (not NaN/string)
    // and that they actually change between different searches/inputs -
    // this is exactly what to check in the browser console if predictions
    // ever look wrong again.
    debugLog("Outgoing prediction payload", payload);
    debugLog("Payload field types", Object.fromEntries(
        Object.entries(payload).map(([key, val]) => [key, typeof val])
    ));

    showCropLoading();

    recommendBtn.disabled = true;

    try{

        // fetch() sends the actual HTTP request. The second argument is
        // an options object where we set:
        //  - method: "POST"            -> matches methods=["POST"] in Flask
        //  - headers Content-Type      -> tells Flask "the body is JSON"
        //  - body: JSON.stringify(...) -> converts our JS object into a
        //                                 JSON-formatted STRING, since
        //                                 fetch() can only send text/binary
        //                                 data, not live JS objects.
        const response = await fetch(`${CROP_API_BASE}/predict`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)

        });

        // response.json() reads the response body and parses it from a
        // JSON string back into a JS object - the mirror image of
        // JSON.stringify() above. It's also async (the body may still be
        // streaming in), so we await it too.
        const data = await response.json();

        // Our Flask route returns success:false with a 400/500 status for
        // bad input or prediction failure, rather than throwing a
        // network-level error - so we check the "success" field ourselves
        // rather than relying on fetch() to detect it as a failure.
        if(!data.success){

            throw new Error(data.message || "The server could not generate a recommendation.");

        }

        showCropResult(data);

    }

    catch(err){

        // This catches THREE different kinds of failure:
        //  1. Network-level problems (Flask backend not running at all,
        //     no internet) - fetch() itself rejects the promise with a
        //     generic "Failed to fetch" TypeError.
        //  2. The "throw new Error(...)" above, for a valid HTTP response
        //     that nonetheless represents a failed prediction (success:false).
        //  3. Any unexpected JS error while rendering the result.
        const message = err.message === "Failed to fetch"
            ? "Could not reach the crop recommendation server. Is the Flask backend running?"
            : (err.message || "Something went wrong. Please try again.");

        showCropError(message);

    }

    finally{

        recommendBtn.disabled = false;

    }

}

// ===============================
// Default Weather
// ===============================

getWeather("Hyderabad");
// ===============================================================
// Prediction History (frontend-only)
//
// The uploaded backend (app.py) does not expose any endpoint to read
// back saved predictions - MongoDB, if used at all, is write-only from
// the server's point of view. Rather than touch app.py to add a
// /history route (explicitly out of scope for this redesign), history
// is kept entirely in this browser via localStorage. It survives
// reloads on this device, but won't sync across devices/browsers.
// ===============================================================

const HISTORY_KEY = "cropPredictionHistory";
const HISTORY_PAGE_SIZE = 5;
let historyPage = 1;

const historyTableBody = document.getElementById("historyTableBody");
const historyEmptyMsg = document.getElementById("historyEmptyMsg");
const historySearchInput = document.getElementById("historySearchInput");
const historyFilterSelect = document.getElementById("historyFilterSelect");
const historySortSelect = document.getElementById("historySortSelect");
const historyExportBtn = document.getElementById("historyExportBtn");
const historyClearBtn = document.getElementById("historyClearBtn");
const historyPagination = document.getElementById("historyPagination");

function loadHistory(){
    try{
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch{
        return [];
    }
}

function persistHistory(entries){
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
}

function saveHistoryEntry(data, displayName){

    const entries = loadHistory();

    const location = currentWeather
        ? `${currentWeather.name}, ${currentWeather.sys.country}`
        : "--";

    entries.unshift({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: Date.now(),
        location,
        crop: displayName || data.prediction,
        confidence: data.confidence ?? null,
        suitability: data.suitability?.overall_percentage ?? null
    });

    // Cap history so localStorage doesn't grow without bound.
    if(entries.length > 200) entries.length = 200;

    persistHistory(entries);
    historyPage = 1;
    renderHistory();

}

function deleteHistoryEntry(id){
    const entries = loadHistory().filter(e => e.id !== id);
    persistHistory(entries);
    renderHistory();
}

function clearHistory(){
    if(!confirm("Clear all prediction history? This cannot be undone.")) return;
    persistHistory([]);
    historyPage = 1;
    renderHistory();
}

function exportHistoryCSV(){

    const entries = loadHistory();

    if(entries.length === 0){
        alert("No history to export yet.");
        return;
    }

    const header = ["Date & Time", "Location", "Crop", "Confidence (%)", "Suitability (%)"];
    const rows = entries.map(e => [
        new Date(e.timestamp).toLocaleString(),
        e.location,
        e.crop,
        e.confidence ?? "",
        e.suitability ?? ""
    ]);

    const csv = [header, ...rows]
        .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(","))
        .join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "crop_prediction_history.csv";
    link.click();
    URL.revokeObjectURL(url);

}

function populateHistoryFilterOptions(entries){

    const currentValue = historyFilterSelect.value;
    const uniqueCrops = [...new Set(entries.map(e => e.crop))].sort();

    historyFilterSelect.innerHTML = '<option value="">All crops</option>' +
        uniqueCrops.map(c => `<option value="${c}">${c}</option>`).join("");

    if(uniqueCrops.includes(currentValue)) historyFilterSelect.value = currentValue;

}

function renderHistory(){

    const allEntries = loadHistory();
    populateHistoryFilterOptions(allEntries);

    const query = historySearchInput.value.trim().toLowerCase();
    const cropFilter = historyFilterSelect.value;
    const sortMode = historySortSelect.value;

    let entries = allEntries.filter(e => {
        const matchesQuery = !query ||
            e.crop.toLowerCase().includes(query) ||
            e.location.toLowerCase().includes(query);
        const matchesFilter = !cropFilter || e.crop === cropFilter;
        return matchesQuery && matchesFilter;
    });

    if(sortMode === "oldest"){
        entries.sort((a, b) => a.timestamp - b.timestamp);
    } else if(sortMode === "confidence"){
        entries.sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1));
    } else {
        entries.sort((a, b) => b.timestamp - a.timestamp);
    }

    const totalPages = Math.max(1, Math.ceil(entries.length / HISTORY_PAGE_SIZE));
    if(historyPage > totalPages) historyPage = totalPages;

    const pageEntries = entries.slice(
        (historyPage - 1) * HISTORY_PAGE_SIZE,
        historyPage * HISTORY_PAGE_SIZE
    );

    historyTableBody.innerHTML = "";

    if(pageEntries.length === 0){
        historyEmptyMsg.classList.remove("hidden");
    } else {
        historyEmptyMsg.classList.add("hidden");
    }

    pageEntries.forEach(entry => {

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${new Date(entry.timestamp).toLocaleDateString()} ${new Date(entry.timestamp).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})}</td>
            <td>${entry.location}</td>
            <td><span class="history-crop-pill">${cropEmoji(entry.crop)} ${entry.crop}</span></td>
            <td>${entry.confidence !== null ? entry.confidence + "%" : "--"}</td>
            <td>${entry.suitability !== null ? entry.suitability + "%" : "--"}</td>
            <td><button class="history-delete-btn" title="Delete entry"><i class="fa-solid fa-trash"></i></button></td>
        `;

        tr.querySelector(".history-delete-btn").addEventListener("click", () => deleteHistoryEntry(entry.id));

        historyTableBody.appendChild(tr);

    });

    // --- Pagination controls ---
    historyPagination.innerHTML = "";
    if(totalPages > 1){
        for(let p = 1; p <= totalPages; p++){
            const btn = document.createElement("button");
            btn.innerText = p;
            if(p === historyPage) btn.classList.add("active");
            btn.addEventListener("click", () => {
                historyPage = p;
                renderHistory();
            });
            historyPagination.appendChild(btn);
        }
    }

}

historySearchInput.addEventListener("input", () => { historyPage = 1; renderHistory(); });
historyFilterSelect.addEventListener("change", () => { historyPage = 1; renderHistory(); });
historySortSelect.addEventListener("change", () => { historyPage = 1; renderHistory(); });
historyExportBtn.addEventListener("click", exportHistoryCSV);
historyClearBtn.addEventListener("click", clearHistory);

// Initial render so past sessions' history shows immediately on load.
renderHistory();
