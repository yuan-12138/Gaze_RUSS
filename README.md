# Gaze_RUSS
## Gaze-Guided Robotic Vascular Ultrasound Leveraging Human Intention Estimation
<div align="center">
<img src=overview.png  width=80%/>
</div>
Overview of the Proposed Gaze-Guided Interactive RUSS: The human gaze signal, captured by a gaze tracker, is combined with segmentation history in an intention estimation module to infer the operator's preference, especially when multiple vessels are visible. The resulting attention heatmap guides the gaze-guided segmentation network to produce accurate vessel segmentation masks. These results are integrated into the robotic control loop to keep the target vessel centered in the ultrasound image. A confidence-based orientation correction optimizes probe contact with curved surfaces, improving image quality. Red boxes in ultrasound images highlight shadowed areas caused by improper probe contact.

<div align="center">
<img src=Gaze_UNet_trans.png  width=80%/>
</div>
(a) The overall design of the proposed gaze-guided segmentation network. (b) The structure of the transformer attention block. (c) The structure of the residual block.

<div align="center">
<img src=Estimation_Module_.png  width=80%/>
</div>
The design of the human intention estimation module.
