import * as THREE from 'three'
import { UIButton, UIInput, UIPanel, UIRow, UISelect, UISpan, UIText } from "./libs/ui.js"
import { AddObjectCommand } from './commands/AddObjectCommand.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

function SideBarGenerationProcess(editor) {
    const config = editor.config
    const strings = editor.strings

    function refreshMeshAlgorithm() {
    }

    const container = new UIPanel()
        .setMarginLeft("-10px")
    const headerRow = new UIRow()
    headerRow.add(new UIText("Processing".toUpperCase()))
    container.add(headerRow)

    function handlePointcloudResponseData(data, filePath, downloadUrl) {
        const geometry = new THREE.BufferGeometry()
        const vertices = new Float32Array(data.flat())
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
        const material = new THREE.PointsMaterial({
            size: 0.03,
            color: 0x0000ff,
            sizeAttenuation: true
        })
        const pointcloud = new THREE.Points(geometry, material)
        pointcloud.name = 'Generated Pointcloud'
        pointcloud.userData.filePath = filePath || null
        pointcloud.userData.downloadUrl = downloadUrl || null
        editor.execute(new AddObjectCommand(editor, pointcloud))
    }

    const buttonRow = new UIRow()
        .setStyle("gap", ["10px"])
        .setMarginTop("10px")

    const upsamplingButton = new UIButton("Upsampling")
    upsamplingButton.onClick(async function() {
        // 🟡 Disable the button and show loading
        upsamplingButton.setTextContent("Upsampling ⏳")
        upsamplingButton.setClass('disabled')
        upsamplingButton.dom.style.pointerEvents = 'none'
        upsamplingButton.dom.style.opacity = '0.5'

        const selected = editor.selected
        if (!selected) {
            alert("No object selected.")
            return
        }
        const url = selected.userData.filePath
        if (!url) {
            alert("Selected object has no file path.")
            return
        }
        const payload = {
            file_path: url,
            file_format: "ply",
            n_points: 8192
        }
        try {
            const response = await fetch("http://localhost:8000/api/inference/upsampling", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            })
            if (response.ok) {
                const data = await response.json()
                console.log("Upsampling response:", data)
                const pointcloudData = data.pointcloud_data
                const filePath = data.file_path
                const downloadUrl = data.download_url
                if (pointcloudData) {
                    handlePointcloudResponseData(pointcloudData, filePath, downloadUrl)
                } else {
                    console.error("No pointcloud data received in response")
                }
            }
        } catch (error) {
            console.error("Error during upsampling:", error)
        } finally {
            // 🟢 Re-enable button after response
            upsamplingButton.setTextContent("Upsampling")
            upsamplingButton.setClass("")
            upsamplingButton.dom.style.pointerEvents = "auto"
            upsamplingButton.dom.style.opacity = "1"
        }
    })
    buttonRow.add(upsamplingButton)

    const meshButton = new UIButton("Generate Mesh")
    meshButton.onClick(async function () {
        // 🟡 Disable the button and show loading
        meshButton.setTextContent("Generating mesh ⏳")
        meshButton.setClass('disabled')
        meshButton.dom.style.pointerEvents = 'none'
        meshButton.dom.style.opacity = '0.5'

        const selected = editor.selected
        if (!selected) {
            alert("No object selected.")
            return
        }
        const filePath = selected.userData.filePath
        const smoothingType = smoothingTypeSelect.getValue()
        if (!filePath) {
            alert("Selected object has no file path.")
            return
        }
        const payload = {
            file_path: filePath,
            smoothing_algorithm: smoothingType, // 'laplacian' or 'taubin'
        }
        try {
            const response = await fetch("http://localhost:8000/api/process/mesh",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                }
            )
            if (!response.ok) {
                throw new Error("Failed to generate mesh");
            }

            // 🟡 Nhận response là blob (file .ply)
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob); // tạo URL tạm để load

            // 🟢 Load PLY từ blob URL
            const loader = new PLYLoader();
            loader.load(blobUrl, function (geometry) {
                geometry.computeVertexNormals();

                const material = new THREE.MeshStandardMaterial({ color: 0xb13e3e });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.name = 'Generated Mesh';

                mesh.userData.downloadUrl = blobUrl || null; // có thể lưu URL tạm này để tải lại nếu cần

                editor.execute(new AddObjectCommand(editor, mesh));
            });
            
        } catch (error) {
            console.error("Error during mesh generation:", error)
        } finally {
            // 🟢 Re-enable button after response
            meshButton.setTextContent("Generate Mesh")
            meshButton.setClass("")
            meshButton.dom.style.pointerEvents = "auto"
            meshButton.dom.style.opacity = "1"
        }
    }) 
    buttonRow.add(meshButton)
    container.add(buttonRow)

    const smoothingRow = new UIRow().setMarginTop("10px")
    container.add(smoothingRow)
    smoothingRow.add(new UIText("Smoothing")).setClass("Label")
    const smoothingTypeSelect = new UISelect()
        .setOptions({
            'laplacian': 'Laplacian',
            'taubin': 'Taubin'
        })
        .setWidth("150px")
        .setMarginLeft("59px")
        .onChange(function () {
            // Handle smoothing type change if needed
        })
        .setValue('laplacian')
    smoothingRow.add(smoothingTypeSelect)

    // const iterationRow = new UIRow()
    // smoothingRow.add(iterationRow)

    // const iterationInput = new UIInput()
    //     .setLeft("100px")
    //     .setWidth("150px")
    // iterationInput.setValue("10")
    // iterationRow.add(
    //     new UIText("Number of iterations")
    //     .setClass("Label")
    // )
    // iterationRow.add(iterationInput)

    return container
}

export { SideBarGenerationProcess }