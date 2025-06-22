import * as THREE from 'three'
import { UIButton, UIInput, UIPanel, UIRow, UISelect, UISpan, UIText } from "./libs/ui.js"
import { AddObjectCommand } from './commands/AddObjectCommand.js';
import { SideBarGenerationProcess } from './Sidebar.Generation.Process.js';

function SidebarGeneration(editor) {
    const config = editor.config;
    const strings = editor.strings

    let selectedFile = null
    const container = new UISpan();

    function handlePointcloudResponseData(data, filePath, downloadUrl) {
        const geometry = new THREE.BufferGeometry()
        const vertices = new Float32Array(data.flat())
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
        const material = new THREE.PointsMaterial({
            size: 0.01,
            color: 0x0ff00,
            sizeAttenuation: true
        })
        const pointcloud = new THREE.Points(geometry, material)
        pointcloud.name = 'Generated Pointcloud'
        pointcloud.userData.filePath = filePath || null
        pointcloud.userData.downloadUrl = downloadUrl || null
        editor.execute(new AddObjectCommand(editor, pointcloud))
    }

    const generation = new UIPanel();
    generation.setBorderTop('0')
    generation.setPaddingTop('20px')
    container.add(generation)

    const headerRow = new UIRow()
    headerRow.add(new UIText('Actions'.toUpperCase()))
    generation.add(headerRow)

    // upload button
    const form = document.createElement('form')
    form.style.display = 'none'
    document.body.appendChild(form)

    const fileInput = document.createElement('input')
    fileInput.multiple = false
    fileInput.type = 'file'
    fileInput.accept = 'image/*'
    fileInput.addEventListener('change', function () {
        const file = event.target.files[0]
        if (file) {
            selectedFile = file
            fileNameInput.setValue(file.name)
            const reader = new FileReader()
            reader.onload = function (e) {
                const image = new Image()
                image.src = e.target.result
                image.onload = function () {
                    imagePreview.dom.src = e.target.result
                    imagePreview.setDisplay('')
                }
            }
            reader.readAsDataURL(file)
            // enable the upload button
            uploadButton.setClass('')
            uploadButton.dom.style.pointerEvents = 'auto'
            uploadButton.dom.style.opacity = '1'
        }
        form.reset()
    })
    form.appendChild(fileInput)

    const buttonRow = new UIRow()
        .setStyle('gap', ['10px'])
    const openButton = new UIButton('Open Image')
    openButton.onClick(function () {
        fileInput.click()
    })
    const uploadButton = new UIButton('Upload Image')
    // disable by default
    uploadButton.setClass('disabled')
    uploadButton.dom.style.pointerEvents = 'none'
    uploadButton.dom.style.opacity = '0.5'
    uploadButton.onClick(async function () {
        if (selectedFile) {
            console.log('Uploading file:', selectedFile.name)

            // 🟡 Disable the button and show loading
            uploadButton.setTextContent('Uploading ⏳')
            uploadButton.setClass('disabled')
            uploadButton.dom.style.pointerEvents = 'none'
            uploadButton.dom.style.opacity = '0.5'

            const formData = new FormData()
            formData.append('file', selectedFile)
            formData.append('file_format', 'ply')

            try {
                const response = await fetch('http://localhost:8000/api/inference/pointcloud', {
                    method: 'POST',
                    body: formData
                })
                if (response.ok) {
                    const data = await response.json()
                    console.log("response: ", data)
                    const pointcloudData = data.pointcloud_data
                    const filePath = data.file_path
                    const downloadUrl = data.download_url
                    if (pointcloudData) {
                        handlePointcloudResponseData(pointcloudData, filePath, downloadUrl)
                    } else {
                        console.error('No pointcloud data received in response')
                    }
                }
            } catch (error) {
                console.error('Error uploading file:', error)
            }
            finally {
                // 🟢 Re-enable button after response
                uploadButton.setTextContent('Upload Image')
                uploadButton.setClass('')
                uploadButton.dom.style.pointerEvents = 'auto'
                uploadButton.dom.style.opacity = '1'
            }
        }
    })
    buttonRow.add(openButton)
    buttonRow.add(uploadButton)
    generation.add(buttonRow)

    // input row
    const inputRow = new UIRow()
        .setMarginTop('10px')
    const fileNameInput = new UIInput('').setWidth('300px')
    fileNameInput.dom.disabled = true
    fileNameInput.setValue('No file selected')

    inputRow.add(fileNameInput)
    generation.add(inputRow)

    const imageRow = new UIRow()
        .setStyle('display', ['flex'])
        .setStyle('justify-content', ['center'])
        .setMarginTop('10px')

    const imagePreview = new UISpan()
    const img = document.createElement('img')
    img.style.maxWidth = '200px'
    img.style.maxHeight = '200px'
    img.style.display = 'none'
    imagePreview.dom = img

    imageRow.add(imagePreview)
    generation.add(imageRow)

    generation.add(new SideBarGenerationProcess(editor))

    return container
}

export { SidebarGeneration}