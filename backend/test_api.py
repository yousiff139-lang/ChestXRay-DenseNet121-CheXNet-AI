import requests
import json

url = "http://localhost:8000/api/v1/classify"

print("\n--- TEST 1: Real Viral Pneumonia Case ---")
with open("frontend/public/samples/real_pneumonia_case.jpg", "rb") as f:
    files = {"file": ("real_pneumonia_case.jpg", f, "image/jpeg")}
    r = requests.post(url, files=files, data={"model": "densenet"})
    d = r.json()
    print("Status:", r.status_code)
    print("Model:", d.get("model_used"))
    print("Disclaimer:", d.get("disclaimer"))
    print("GradCAM Class:", d.get("gradcam_class_name"))
    print("Has GradCAM Overlay:", d.get("gradcam_overlay_base64") is not None)
    print("Inference ms:", d.get("inference_time_ms"))
    print("Model pass ms:", d.get("model_inference_ms"))
    print("End to End ms:", d.get("end_to_end_ms"))
    print("Findings:")
    for p in d.get("positive_findings", []):
        print(f"  * {p['disease_name']:<18}: {p['probability']*100:.1f}% | State: {p.get('state')} | Cutoff: {p.get('threshold', 0)*100:.1f}% | Prevalence: {(p.get('prevalence') or 0)*100:.1f}%")

print("\n--- TEST 2: Real Normal Control Case ---")
with open("frontend/public/samples/real_normal_case.jpg", "rb") as f:
    files = {"file": ("real_normal_case.jpg", f, "image/jpeg")}
    r = requests.post(url, files=files, data={"model": "densenet"})
    d = r.json()
    print("Status:", r.status_code)
    print("GradCAM Class:", d.get("gradcam_class_name"))
    print("Has GradCAM Overlay:", d.get("gradcam_overlay_base64") is not None)
    print("Findings Count:", len(d.get("positive_findings", [])))
    for p in d.get("positive_findings", []):
        print(f"  * {p['disease_name']:<18}: {p['probability']*100:.1f}% | State: {p.get('state')}")
