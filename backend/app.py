import os
import re
import sys
import subprocess
import joblib
import requests
import tldextract
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, unquote
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
TRAIN_SCRIPT = os.path.join(MODEL_DIR, "train_model.py")

MODEL_PATHS = {
    "rf": os.path.join(MODEL_DIR, "phishing_live_model.pkl"),
    "lr": os.path.join(MODEL_DIR, "phishing_lr_model.pkl"),
    "lr_scaler": os.path.join(MODEL_DIR, "lr_scaler.pkl"),
    "feature_names": os.path.join(MODEL_DIR, "feature_names.pkl"),
}

def _run_training_script() -> None:
    """Rebuild model artifacts for the currently installed sklearn version."""
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=MODEL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Model retraining failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

def _load_model_artifacts():
    model_rf = joblib.load(MODEL_PATHS["rf"])
    model_lr = joblib.load(MODEL_PATHS["lr"])
    lr_scaler = joblib.load(MODEL_PATHS["lr_scaler"])
    feature_names = joblib.load(MODEL_PATHS["feature_names"])

    models = {
        "random_forest": {"model": model_rf, "label": "Random Forest", "scaler": None},
        "logistic_regression": {"model": model_lr, "label": "Logistic Regression", "scaler": lr_scaler},
    }
    return models, feature_names

def _load_or_rebuild_models():
    try:
        return _load_model_artifacts()
    except Exception as exc:
        print(f"Model loading failed ({exc}). Rebuilding model files...")
        _run_training_script()
        return _load_model_artifacts()

MODELS, feature_names = _load_or_rebuild_models()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', '.online', '.site', '.space']
# Suspicious keywords in domain
SUSPICIOUS_KEYWORDS = ['secure', 'verify', 'login', 'signin', 'account', 'update', 'confirm', 'banking', 'paypal', 'apple', 'microsoft', 'amazon']

def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url

def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)

def count_letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)

def count_special(text: str) -> int:
    return sum(not ch.isalnum() for ch in text)

def is_ip_domain(domain: str) -> int:
    return 1 if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", domain) else 0

def check_suspicious_patterns(url: str, domain: str) -> dict:
    """Check for suspicious patterns that indicate phishing"""
    suspicious_score = 0
    reasons = []
    
    # Check TLD
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            suspicious_score += 2
            reasons.append(f"Suspicious TLD: {tld}")
            break
    
    # Check for suspicious keywords
    domain_lower = domain.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in domain_lower:
            suspicious_score += 1
            reasons.append(f"Suspicious keyword: {keyword}")
    
    # Check for excessive subdomains
    subdomain_count = len(domain.split('.')) - 2
    if subdomain_count > 2:
        suspicious_score += subdomain_count
        reasons.append(f"Excessive subdomains: {subdomain_count}")
    
    # Check for IP address in URL
    if is_ip_domain(domain):
        suspicious_score += 3
        reasons.append("IP address used instead of domain")
    
    # Check for HTTP (not HTTPS)
    if not url.startswith("https://"):
        suspicious_score += 1
        reasons.append("No HTTPS (insecure connection)")
    
    return {
        "suspicious_score": suspicious_score,
        "suspicious_reasons": reasons,
        "is_suspicious": suspicious_score >= 2
    }

def url_features(url: str) -> dict:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    ext = tldextract.extract(url)

    decoded_url = unquote(url)
    url_len = len(url)
    domain_len = len(domain)
    tld_len = len(ext.suffix)

    subdomain_count = 0
    if ext.subdomain:
        subdomain_count = len([x for x in ext.subdomain.split(".") if x])

    obfuscated_chars = url.count("%")
    has_obfuscation = 1 if "%" in url else 0
    obfuscation_ratio = obfuscated_chars / url_len if url_len else 0

    letter_count = count_letters(url)
    digit_count = count_digits(url)
    special_count = count_special(url)
    
    # Extract query parameters
    query = parsed.query
    equals_count = query.count("=")
    qmark_count = 1 if parsed.query else 0
    ampersand_count = query.count("&")
    
    # Check suspicious patterns
    suspicious = check_suspicious_patterns(url, domain)

    return {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "IsDomainIP": is_ip_domain(domain),
        "TLDLength": tld_len,
        "NoOfSubDomain": subdomain_count,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letter_count,
        "LetterRatioInURL": letter_count / url_len if url_len else 0,
        "NoOfDegitsInURL": digit_count,
        "DegitRatioInURL": digit_count / url_len if url_len else 0,
        "NoOfEqualsInURL": equals_count,
        "NoOfQMarkInURL": qmark_count,
        "NoOfAmpersandInURL": ampersand_count,
        "NoOfOtherSpecialCharsInURL": special_count,
        "SpacialCharRatioInURL": special_count / url_len if url_len else 0,
        "IsHTTPS": 1 if parsed.scheme.lower() == "https" else 0,
        "_suspicious_score": suspicious["suspicious_score"],
        "_suspicious_reasons": suspicious["suspicious_reasons"],
        "_is_suspicious": suspicious["is_suspicious"]
    }

def page_features(url: str) -> dict:
    default_features = {
        "LineOfCode": 0, "LargestLineLength": 0, "HasTitle": 0, "HasFavicon": 0,
        "Robots": 0, "IsResponsive": 0, "NoOfURLRedirect": 0, "NoOfSelfRedirect": 0,
        "HasDescription": 0, "NoOfPopup": 0, "NoOfiFrame": 0, "HasExternalFormSubmit": 0,
        "HasSocialNet": 0, "HasSubmitButton": 0, "HasHiddenFields": 0, "HasPasswordField": 0,
        "Bank": 0, "Pay": 0, "Crypto": 0, "HasCopyrightInfo": 0, "NoOfImage": 0,
        "NoOfCSS": 0, "NoOfJS": 0, "NoOfSelfRef": 0, "NoOfEmptyRef": 0, "NoOfExternalRef": 0
    }
    
    try:
        response = requests.get(url, timeout=10, headers=HEADERS, allow_redirects=True)
        response.raise_for_status()

        final_url = response.url
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        parsed_final = urlparse(final_url)
        base_domain = parsed_final.netloc.lower()

        # Basic HTML structure
        lines = html.splitlines()
        line_of_code = len(lines)
        largest_line_length = max((len(line) for line in lines), default=0)
        
        # Meta tags
        title_tag = soup.find("title")
        has_title = 1 if title_tag and title_tag.get_text(strip=True) else 0
        
        # Favicon
        has_favicon = 0
        for link in soup.find_all("link", href=True):
            rel = " ".join([str(x).lower() for x in link.get("rel", [])])
            href = link.get("href", "").lower()
            if "icon" in rel or "favicon" in href:
                has_favicon = 1
                break
        
        # Meta tags
        robots = 1 if soup.find("meta", attrs={"name": re.compile(r"robots", re.I)}) else 0
        responsive = 1 if soup.find("meta", attrs={"name": re.compile(r"viewport", re.I)}) else 0
        has_description = 1 if soup.find("meta", attrs={"name": re.compile(r"description", re.I)}) else 0
        
        # Scripts and popups
        no_of_popup = html.lower().count("window.open")
        no_of_iframe = len(soup.find_all("iframe"))
        
        # Forms analysis
        forms = soup.find_all("form")
        has_external_form_submit = 0
        has_submit_button = 0
        has_hidden_fields = 0
        has_password_field = 0
        
        for form in forms:
            action = form.get("action", "").strip()
            if action:
                abs_action = urljoin(final_url, action)
                action_domain = urlparse(abs_action).netloc.lower()
                if action_domain and action_domain != base_domain:
                    has_external_form_submit = 1
            
            if form.find("input", attrs={"type": re.compile(r"submit", re.I)}) or \
               form.find("button", attrs={"type": re.compile(r"submit", re.I)}):
                has_submit_button = 1
            
            if form.find("input", attrs={"type": re.compile(r"hidden", re.I)}):
                has_hidden_fields = 1
            
            if form.find("input", attrs={"type": re.compile(r"password", re.I)}):
                has_password_field = 1
        
        # Social networks
        social_keywords = ["facebook.com", "twitter.com", "instagram.com", 
                          "linkedin.com", "youtube.com", "t.me", "telegram"]
        has_social = 0
        
        # Links analysis
        links = soup.find_all("a", href=True)
        self_ref = 0
        empty_ref = 0
        external_ref = 0
        
        for a in links:
            href = a.get("href", "").strip()
            
            if not href or href == "#" or href.lower().startswith("javascript:"):
                empty_ref += 1
                continue
            
            abs_url = urljoin(final_url, href)
            href_domain = urlparse(abs_url).netloc.lower()
            
            if any(sk in abs_url.lower() for sk in social_keywords):
                has_social = 1
            
            if not href_domain or href_domain == base_domain:
                self_ref += 1
            else:
                external_ref += 1
        
        # Count images, CSS, JS
        images = len(soup.find_all("img"))
        
        css_files = 0
        for link in soup.find_all("link", href=True):
            rel = " ".join([str(x).lower() for x in link.get("rel", [])])
            href = link.get("href", "").lower()
            if "stylesheet" in rel or href.endswith(".css"):
                css_files += 1
        
        js_files = len(soup.find_all("script"))
        
        # Text analysis
        text = soup.get_text(" ", strip=True).lower()
        html_lower = html.lower()
        
        # Keywords
        bank = 1 if "bank" in text or "bank" in final_url.lower() else 0
        pay = 1 if "pay" in text or "payment" in text or "pay" in final_url.lower() else 0
        crypto = 1 if "crypto" in text or "bitcoin" in text or "crypto" in final_url.lower() else 0
        has_copyright = 1 if "copyright" in text or "©" in html else 0
        
        return {
            "LineOfCode": line_of_code,
            "LargestLineLength": largest_line_length,
            "HasTitle": has_title,
            "HasFavicon": has_favicon,
            "Robots": robots,
            "IsResponsive": responsive,
            "NoOfURLRedirect": len(response.history),
            "NoOfSelfRedirect": 0,
            "HasDescription": has_description,
            "NoOfPopup": no_of_popup,
            "NoOfiFrame": no_of_iframe,
            "HasExternalFormSubmit": has_external_form_submit,
            "HasSocialNet": has_social,
            "HasSubmitButton": has_submit_button,
            "HasHiddenFields": has_hidden_fields,
            "HasPasswordField": has_password_field,
            "Bank": bank,
            "Pay": pay,
            "Crypto": crypto,
            "HasCopyrightInfo": has_copyright,
            "NoOfImage": images,
            "NoOfCSS": css_files,
            "NoOfJS": js_files,
            "NoOfSelfRef": self_ref,
            "NoOfEmptyRef": empty_ref,
            "NoOfExternalRef": external_ref,
            "_final_url": final_url,
            "_fetch_status": "success",
        }
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        default_features["_final_url"] = url
        default_features["_fetch_status"] = f"failed: {str(e)[:100]}"
        return default_features

def extract_live_features(url: str) -> dict:
    url = normalize_url(url)
    f1 = url_features(url)
    f2 = page_features(url)
    
    combined = {}
    combined.update(f1)
    for k, v in f2.items():
        if not k.startswith("_"):
            combined[k] = v
    
    combined["_final_url"] = f2.get("_final_url", url)
    combined["_fetch_status"] = f2.get("_fetch_status", "unknown")
    combined["_suspicious_score"] = f1.get("_suspicious_score", 0)
    combined["_suspicious_reasons"] = f1.get("_suspicious_reasons", [])
    combined["_is_suspicious"] = f1.get("_is_suspicious", False)
    
    return combined

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Phishing detection API",
        "models": list(MODELS.keys()),
        "features_count": len(feature_names)
    })

@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.get_json()
        url = data.get("url", "").strip() if data else ""
        
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        
        features = extract_live_features(url)
        
        # Return only the features needed for the model
        model_features = {}
        for col in feature_names:
            model_features[col] = features.get(col, 0)
        
        return jsonify({
            "model": "feature_extraction",
            "url": features["_final_url"],
            "fetch_status": features["_fetch_status"],
            "suspicious_score": features.get("_suspicious_score", 0),
            "suspicious_reasons": features.get("_suspicious_reasons", []),
            "features": model_features
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "").strip() if data else ""
        model_key = data.get("model", "random_forest") if data else "random_forest"

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        if model_key not in MODELS:
            return jsonify({"error": f"Unknown model '{model_key}'. Choose from: {list(MODELS.keys())}"}), 400

        selected = MODELS[model_key]
        active_model = selected["model"]
        MODEL_LABEL = selected["label"]
        scaler = selected["scaler"]

        features = extract_live_features(url)

        # Prepare feature vector
        X = pd.DataFrame([{col: features.get(col, 0) for col in feature_names}])
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feature_names)

        # Predict
        prediction_raw = int(active_model.predict(X)[0])

        # Get confidence
        confidence = None
        if hasattr(active_model, "predict_proba"):
            confidence = float(active_model.predict_proba(X)[0].max()) * 100
        
        # Check if fetch failed
        fetch_status = features.get("_fetch_status", "unknown")
        fetch_failed = "failed" in str(fetch_status).lower()
        is_suspicious = features.get("_is_suspicious", False)
        suspicious_score = features.get("_suspicious_score", 0)
        suspicious_reasons = features.get("_suspicious_reasons", [])
        
        # OVERRIDE LOGIC: If fetch failed OR URL is suspicious, mark as phishing
        warning = None
        final_prediction = prediction_raw
        
        if fetch_failed:
            warning = "⚠️ Could not fetch webpage - cannot verify content"
            # If fetch failed but model says legitimate, override to suspicious
            if final_prediction == 0:
                final_prediction = 1
                confidence = 50.0 if confidence else 50.0
        elif is_suspicious and final_prediction == 0:
            warning = f"⚠️ URL shows suspicious patterns (score: {suspicious_score})"
            # Reduce confidence but keep model prediction
            confidence = confidence * 0.7 if confidence else 70.0
        
        label = "Legitimate" if final_prediction == 0 else "Phishing"
        
        response_data = {
            "model": MODEL_LABEL,
            "model_key": model_key,
            "url": features["_final_url"],
            "fetch_status": fetch_status,
            "prediction": final_prediction,
            "label": label,
            "confidence": round(confidence, 2) if confidence else None,
            "suspicious_score": suspicious_score,
            "suspicious_reasons": suspicious_reasons,
            "used_features": {col: features.get(col, 0) for col in feature_names}
        }
        
        if warning:
            response_data["warning"] = warning
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No feature data provided"}), 400
        
        model_key = data.get("model", "random_forest")
        if model_key not in MODELS:
            return jsonify({"error": f"Unknown model '{model_key}'. Choose from: {list(MODELS.keys())}"}), 400

        selected = MODELS[model_key]
        active_model = selected["model"]
        model_label = selected["label"]
        scaler = selected["scaler"]

        # Ensure all features are present
        X = pd.DataFrame([{col: data.get(col, 0) for col in feature_names}])
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feature_names)
        
        prediction_raw = int(active_model.predict(X)[0])
        
        confidence = None
        if hasattr(active_model, "predict_proba"):
            confidence = float(active_model.predict_proba(X)[0].max()) * 100
        
        label = "Legitimate" if prediction_raw == 0 else "Phishing"
        
        return jsonify({
            "model": model_label,
            "model_key": model_key,
            "fetch_status": "manual input",
            "prediction": prediction_raw,
            "label": label,
            "confidence": round(confidence, 2) if confidence else None,
            "used_features": {col: data.get(col, 0) for col in feature_names}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)