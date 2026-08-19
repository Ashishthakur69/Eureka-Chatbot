document.addEventListener("DOMContentLoaded", () => {

    document.body.classList.remove("loading");

    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const fileList = document.getElementById("file-list");

    const themeToggle = document.getElementById("theme-toggle");
    const themeIcon = document.getElementById("theme-icon");
    const themeText = document.getElementById("theme-text");

    let uploadedDocuments = [];


    // Add a message to the chat.
    function addMessage(sender, text) {

        const messageDiv = document.createElement("div");

        messageDiv.className = `message ${sender}`;

        const textElement = document.createElement("div");

        textElement.className = "message-text";

        if (sender === "bot") {
            textElement.innerHTML = marked.parse(text);
        } else {
            textElement.textContent = text;
        }

        messageDiv.appendChild(textElement);

        if (sender === "bot") {

            const copyButton =
                document.createElement("i");

            copyButton.className =
                "fas fa-copy copy-btn";

            copyButton.title =
                "Copy text";

            messageDiv.appendChild(
                copyButton
            );
        }

        chatBox.appendChild(messageDiv);

        chatBox.scrollTop =
            chatBox.scrollHeight;
    }


    // Send a message to the backend.
    async function sendMessage(message) {

        if (!message.trim()) {
            return;
        }

        addMessage("user", message);

        userInput.value = "";

        const messageDiv =
            document.createElement("div");

        messageDiv.className =
            "message bot";

        const textElement =
            document.createElement("div");

        textElement.className =
            "message-text";

        messageDiv.appendChild(
            textElement
        );

        chatBox.appendChild(
            messageDiv
        );

        sendBtn.disabled = true;
        userInput.disabled = true;

        try {

            const response = await fetch(
                "/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })
                }
            );

            if (!response.ok) {
                throw new Error(
                    `Server error: ${response.status}`
                );
            }

            if (!response.body) {
                throw new Error(
                    "No response received."
                );
            }

            const reader =
                response.body.getReader();

            const decoder =
                new TextDecoder();

            let fullText = "";

            while (true) {

                const {
                    value,
                    done
                } = await reader.read();

                if (done) {
                    break;
                }

                fullText += decoder.decode(
                    value,
                    {
                        stream: true
                    }
                );

                textElement.innerHTML =
                    marked.parse(
                        fullText + "▋"
                    );

                chatBox.scrollTop =
                    chatBox.scrollHeight;
            }

            textElement.innerHTML =
                marked.parse(fullText);

            // Add copy button after response.
            const copyButton =
                document.createElement("i");

            copyButton.className =
                "fas fa-copy copy-btn";

            copyButton.title =
                "Copy text";

            messageDiv.appendChild(
                copyButton
            );

        } catch (error) {

            console.error(
                "Chat error:",
                error
            );

            textElement.textContent =
                "❌ Could not connect to Eureka.";

        } finally {

            sendBtn.disabled = false;
            userInput.disabled = false;

            userInput.focus();
        }
    }


    // Update the document counter.
    function updateDocumentCount() {

        const counter =
            document.getElementById(
                "document-count"
            );

        if (!counter) {
            return;
        }

        counter.textContent =
            `${uploadedDocuments.length} / 5`;
    }


    // Display all uploaded documents.
    function renderDocuments() {

        fileList.innerHTML = "";

        if (uploadedDocuments.length === 0) {

            const emptyMessage =
                document.createElement("p");

            emptyMessage.className =
                "no-files";

            emptyMessage.textContent =
                "📪 No documents uploaded yet.";

            fileList.appendChild(
                emptyMessage
            );

            updateDocumentCount();

            return;
        }


        uploadedDocuments.forEach(
            (filename) => {

                const fileItem =
                    document.createElement("div");

                fileItem.className =
                    "file-item";


                const fileInfo =
                    document.createElement("div");

                fileInfo.className =
                    "file-item-info";


                const fileIcon =
                    document.createElement("i");

                fileIcon.className =
                    "fas fa-file-alt";


                const fileName =
                    document.createElement("span");

                fileName.className =
                    "file-name";

                fileName.textContent =
                    filename;


                fileInfo.appendChild(
                    fileIcon
                );

                fileInfo.appendChild(
                    fileName
                );


                // Delete button.
                const deleteButton =
                    document.createElement("button");

                deleteButton.type =
                    "button";

                deleteButton.className =
                    "delete-btn";

                deleteButton.title =
                    `Delete ${filename}`;

                deleteButton.dataset.filename =
                    filename;

                deleteButton.innerHTML =
                    '<i class="fas fa-trash"></i>';


                fileItem.appendChild(
                    fileInfo
                );

                fileItem.appendChild(
                    deleteButton
                );

                fileList.appendChild(
                    fileItem
                );
            }
        );

        updateDocumentCount();
    }


    // Load documents from the backend.
    async function loadDocuments() {

        try {

            const response =
                await fetch(
                    "/documents"
                );

            if (!response.ok) {
                return;
            }

            const data =
                await response.json();

            uploadedDocuments =
                data.documents || [];

            renderDocuments();

        } catch (error) {

            console.error(
                "Document loading error:",
                error
            );
        }
    }


    // Upload a document.
    async function uploadFile(file) {

        const temporaryItem =
            document.createElement("div");

        temporaryItem.className =
            "file-item";


        const fileInfo =
            document.createElement("div");

        fileInfo.className =
            "file-item-info";


        const icon =
            document.createElement("i");

        icon.className =
            "fas fa-file-alt";


        const name =
            document.createElement("span");

        name.className =
            "file-name";

        name.textContent =
            file.name;


        fileInfo.appendChild(icon);
        fileInfo.appendChild(name);


        const status =
            document.createElement("span");

        status.className =
            "file-status processing";

        status.textContent =
            "Processing...";


        temporaryItem.appendChild(
            fileInfo
        );

        temporaryItem.appendChild(
            status
        );

        fileList.appendChild(
            temporaryItem
        );


        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    "/upload",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error ||
                    "Upload failed."
                );
            }


            uploadedDocuments =
                data.documents || [];

            renderDocuments();


        } catch (error) {

            console.error(
                "Upload error:",
                error
            );

            temporaryItem.remove();

            renderDocuments();

            alert(
                error.message ||
                "Failed to upload document."
            );
        }
    }


    // Handle selected files.
    function handleFiles(files) {

        if (!files || files.length === 0) {
            return;
        }


        const availableSlots =
            5 - uploadedDocuments.length;


        if (availableSlots <= 0) {

            alert(
                "You can upload a maximum of 5 documents."
            );

            return;
        }


        const selectedFiles =
            Array.from(files).slice(
                0,
                availableSlots
            );


        selectedFiles.forEach(
            (file) => {

                const allowedTypes = [
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ];


                if (
                    !allowedTypes.includes(
                        file.type
                    )
                ) {

                    alert(
                        `${file.name} is not supported. Please upload a PDF or DOCX file.`
                    );

                    return;
                }


                if (
                    uploadedDocuments.includes(
                        file.name
                    )
                ) {

                    alert(
                        `${file.name} is already uploaded.`
                    );

                    return;
                }


                uploadFile(file);
            }
        );


        fileInput.value = "";
    }


    // Delete one document.
    async function deleteDocument(filename) {

        const confirmed =
            confirm(
                `Delete "${filename}"?`
            );


        if (!confirmed) {
            return;
        }


        try {

            const response =
                await fetch(
                    "/delete_document",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            filename: filename
                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error ||
                    "Could not delete document."
                );
            }


            uploadedDocuments =
                data.documents || [];


            renderDocuments();


        } catch (error) {

            console.error(
                "Delete error:",
                error
            );

            alert(
                error.message ||
                "Failed to delete document."
            );
        }
    }


    // Upload zone click.
    uploadZone.addEventListener(
        "click",
        () => fileInput.click()
    );


    // Drag over.
    uploadZone.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            uploadZone.classList.add(
                "dragover"
            );
        }
    );


    // Drag leave.
    uploadZone.addEventListener(
        "dragleave",
        () => {

            uploadZone.classList.remove(
                "dragover"
            );
        }
    );


    // Drop files.
    uploadZone.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            uploadZone.classList.remove(
                "dragover"
            );

            handleFiles(
                event.dataTransfer.files
            );
        }
    );


    // File picker.
    fileInput.addEventListener(
        "change",
        () => {

            handleFiles(
                fileInput.files
            );
        }
    );


    // Send message.
    sendBtn.addEventListener(
        "click",
        () => {

            sendMessage(
                userInput.value
            );
        }
    );


    // Enter to send.
    userInput.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage(
                    userInput.value
                );
            }
        }
    );


    // Handle delete and copy buttons.
    document.addEventListener(
        "click",
        async (event) => {

            const deleteButton =
                event.target.closest(
                    ".delete-btn"
                );


            if (deleteButton) {

                const filename =
                    deleteButton.dataset.filename;


                if (filename) {

                    await deleteDocument(
                        filename
                    );
                }

                return;
            }


            const copyButton =
                event.target.closest(
                    ".copy-btn"
                );


            if (copyButton) {

                const message =
                    copyButton
                        .closest(".message")
                        ?.querySelector(
                            ".message-text"
                        );


                if (!message) {
                    return;
                }


                try {

                    await navigator.clipboard.writeText(
                        message.innerText
                    );


                    copyButton.classList.replace(
                        "fa-copy",
                        "fa-check"
                    );


                    setTimeout(
                        () => {

                            copyButton.classList.replace(
                                "fa-check",
                                "fa-copy"
                            );

                        },
                        1500
                    );

                } catch (error) {

                    console.error(
                        "Copy failed:",
                        error
                    );
                }
            }
        }
    );


    // Theme switch.
    themeToggle.addEventListener(
        "change",
        () => {

            document.body.classList.toggle(
                "light-theme"
            );


            if (themeToggle.checked) {

                themeIcon.className =
                    "fas fa-moon";

                themeText.textContent =
                    "Dark Mode";

            } else {

                themeIcon.className =
                    "fas fa-sun";

                themeText.textContent =
                    "Light Mode";
            }
        }
    );


    // Initial message.
    addMessage(
        "bot",
        "Hello! I'm **Eureka** 💡. Upload documents or ask me anything."
    );


    // Load existing documents.
    loadDocuments();

});