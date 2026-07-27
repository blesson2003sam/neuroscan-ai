@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint — receives MRI image, returns:
    - Predicted class
    - Confidence scores for all 4 classes
    - Grad-CAM heatmap images (base64)
    """
    # ── Validate file ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    # ── Load image ──
    contents = await file.read()
    original_img = Image.open(BytesIO(contents)).convert("RGB")
    original_img_resized = original_img.resize((224, 224))
    img_array = np.array(original_img_resized)

    # ── Preprocess ──
    img_tensor = transform(original_img_resized).unsqueeze(0)

    # ── Predict ──
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)

    # Get all confidence scores
    confidences = {}
    for i, cls in enumerate(CLASS_NAMES):
        confidences[cls] = round(float(probabilities[0][i]) * 100, 2)

    # Get top prediction
    predicted_idx = probabilities.argmax(1).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = confidences[predicted_class]

    # ── Generate Grad-CAM ──
    img_tensor_grad = transform(original_img_resized).unsqueeze(0)
    img_tensor_grad.requires_grad_(True)
    cam = gradcam.generate(img_tensor_grad, predicted_idx)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_rgb, 0.4, 0)

    # ── Return results ──
    return {
        "success": True,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence,
            "confidences": confidences,
            "info": CLASS_INFO[predicted_class]
        },
        "images": {
            "original": image_to_base64(img_array),
            "heatmap": image_to_base64(heatmap_rgb),
            "overlay": image_to_base64(overlay)
        }
    }
