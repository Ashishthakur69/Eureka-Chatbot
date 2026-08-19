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

    let documents = [];

    const MAX_DOCUMENTS = 5;


    // Create a message element

    function createMessageDiv(sender) {

        const messageDiv =
            document.createElement("div");

        messageDiv.className =
            `message ${sender}`;

        const textElement =
            document.createElement("div");

        textElement.className =
            "message-text";

        messageDiv.appendChild(
            textElement
        );

        return messageDiv;
    }


    // Add a message to the chat

    function addMessage(sender, text) {

        const messageDiv =
            createMessageDiv(sender);

        const textElement =
            messageDiv.querySelector(
                ".message-text"
            );

        if (sender === "bot") {

            textElement.innerHTML =
                marked.parse(text);

            const copyBtn =
                document.createElement("i");

            copyBtn.className =
                "fas fa-copy copy-btn";

            copyBtn.title =
                "Copy text";

            messageDiv.appendChild(
                copyBtn
            );

        } else {

            textElement.textContent =
                text;
        }

        chatBox.appendChild(
            messageDiv
        );

        chatBox.scrollTop =
            chatBox.scrollHeight;
    }


    // Send a chat message

    async function sendMessage(message) {

        if (!message.trim()) {
            return;
        }

        addMessage(
            "user",
            message
        );

        userInput.value = "";

        const botMessage =
            createMessageDiv("bot");

        const textElement =
            botMessage.querySelector(
                ".message-text"
            );

        chatBox.appendChild(
            botMessage
        );

        chatBox.scrollTop =
            chatBox.scrollHeight;

        try {

            const response =
                await fetch(
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
                    `Server returned ${response.status}`
                );
            }

            if (!response.body) {

                throw new Error(
                    "No response body."
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

                fullText +=
                    decoder.decode(
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
                marked.parse(
                    fullText
                );

            addCopyButton(
                botMessage
            );

        } catch (error) {

            console.error(
                "Chat error:",
                error
            );

            textElement.innerHTML =
                marked.parse(
                    "❌ Could not connect to Eureka."
                );
        }
    }


    // Add copy button

    function addCopyButton(
        messageDiv
    ) {

        if (
            messageDiv.querySelector(
                ".copy-btn"
            )
        ) {
            return;
        }

        const copyBtn =
            document.createElement("i");

        copyBtn.className =
            "fas fa-copy copy-btn";

        copyBtn.title =
            "Copy text";

        messageDiv.appendChild(
            copyBtn
        );
    }


    // Update document counter

    function updateDocumentCount() {

        const count =
            documents.length;

        const counter =
            document.querySelector(
                ".document-count"
            );

        if (counter) {

            counter.textContent =
                `${count} / ${MAX_DOCUMENTS}`;
        }
    }


    // Show empty document message

    function showEmptyMessage() {

        const existingItems =
            fileList.querySelectorAll(
                ".file-item"
            );

        existingItems.forEach(
            item => item.remove()
        );

        let noFiles =
            fileList.querySelector(
                ".no-files"
            );

        if (!noFiles) {

            noFiles =
                document.createElement("p");

            noFiles.className =
                "no-files";

            fileList.appendChild(
                noFiles
            );
        }

        noFiles.textContent =
            "📪 No documents uploaded yet.";

        noFiles.style.display =
            "block";
    }


    // Hide empty document message

    function hideEmptyMessage() {

        const noFiles =
            fileList.querySelector(
                ".no-files"
            );

        if (noFiles) {

            noFiles.style.display =
                "none";
        }
    }


    // Create document row

    function createDocumentElement(
        documentData
    ) {

        const fileItem =
            document.createElement("div");

        fileItem.className =
            "file-item";

        fileItem.dataset.documentId =
            documentData.document_id;

        const info =
            document.createElement("div");

        info.className =
            "file-item-info";

        const fileIcon =
            document.createElement("i");

        fileIcon.className =
            "fas fa-file-pdf";

        if (
            documentData.filename
                .toLowerCase()
                .endsWith(".docx")
        ) {

            fileIcon.className =
                "fas fa-file-word";
        }

        const fileName =
            document.createElement("span");

        fileName.className =
            "file-name";

        fileName.textContent =
            documentData.filename;

        fileName.title =
            documentData.filename;

        info.appendChild(
            fileIcon
        );

        info.appendChild(
            fileName
        );


        const status =
            document.createElement("span");

        status.className =
            "file-status";

        status.textContent =
            "Ready ✓";


        const deleteButton =
            document.createElement("button");

        deleteButton.className =
            "delete-btn";

        deleteButton.type =
            "button";

        deleteButton.title =
            "Delete document";

        deleteButton.dataset.documentId =
            documentData.document_id;

        deleteButton.innerHTML =
            '<i class="fas fa-trash"></i>';


        fileItem.appendChild(
            info
        );

        fileItem.appendChild(
            status
        );

        fileItem.appendChild(
            deleteButton
        );

        return fileItem;
    }


    // Render all documents

    function renderDocuments() {

        fileList.innerHTML = "";

        if (
            documents.length === 0
        ) {

            showEmptyMessage();

            updateDocumentCount();

            return;
        }

        hideEmptyMessage();

        documents.forEach(
            documentData => {

                const element =
                    createDocumentElement(
                        documentData
                    );

                fileList.appendChild(
                    element
                );
            }
        );

        updateDocumentCount();
    }


    // Load existing documents

    async function loadDocuments() {

        try {

            const response =
                await fetch(
                    "/documents"
                );

            if (!response.ok) {

                throw new Error(
                    "Could not load documents."
                );
            }

            const data =
                await response.json();

            documents =
                data.documents || [];

            renderDocuments();

        } catch (error) {

            console.error(
                "Loading documents failed:",
                error
            );

            documents = [];

            renderDocuments();
        }
    }


    // Handle selected files

    function handleFiles(files) {

        if (!files || files.length === 0) {
            return;
        }

        if (
            documents.length >=
            MAX_DOCUMENTS
        ) {

            alert(
                "You can upload up to 5 documents."
            );

            return;
        }

        if (files.length > 1) {

            alert(
                "Please upload one document at a time."
            );

            return;
        }

        const file = files[0];

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
                "Unsupported file type. Please upload a PDF or DOCX file."
            );

            return;
        }

        uploadFile(file);
    }


    // Upload document

    async function uploadFile(file) {

        const temporaryItem =
            document.createElement("div");

        temporaryItem.className =
            "file-item";

        temporaryItem.innerHTML = `
            <div class="file-item-info">
                <i class="fas fa-file"></i>
                <span class="file-name" title="${escapeHtml(file.name)}">
                    ${escapeHtml(file.name)}
                </span>
            </div>

            <span class="file-status processing">
                Processing...
            </span>

            <button
                class="delete-btn"
                type="button"
                disabled
                title="Processing"
            >
                <i class="fas fa-spinner fa-spin"></i>
            </button>
        `;

        hideEmptyMessage();

        fileList.appendChild(
            temporaryItem
        );

        updateDocumentCount();

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

            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    "Upload failed."
                );
            }

            temporaryItem.remove();

            documents.push({

                filename:
                    data.filename,

                document_id:
                    data.document_id,

                chunks:
                    data.chunks

            });

            renderDocuments();

            console.log(
                "Document uploaded:",
                data
            );

        } catch (error) {

            console.error(
                "Upload error:",
                error
            );

            temporaryItem.remove();

            renderDocuments();

            alert(
                `Upload failed: ${error.message}`
            );
        }
    }


    // Delete one document

    async function deleteDocument(
        documentId,
        button
    ) {

        const documentData =
            documents.find(
                doc =>
                    doc.document_id ===
                    documentId
            );

        if (!documentData) {
            return;
        }

        const confirmed =
            confirm(
                `Delete "${documentData.filename}"?`
            );

        if (!confirmed) {
            return;
        }

        button.disabled =
            true;

        button.innerHTML =
            '<i class="fas fa-spinner fa-spin"></i>';

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
                            document_id:
                                documentId
                        })
                    }
                );

            const data =
                await response.json();

            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    "Could not delete document."
                );
            }

            documents =
                documents.filter(
                    doc =>
                        doc.document_id !==
                        documentId
                );

            renderDocuments();

        } catch (error) {

            console.error(
                "Delete error:",
                error
            );

            button.disabled =
                false;

            button.innerHTML =
                '<i class="fas fa-trash"></i>';

            alert(
                `Could not delete document: ${error.message}`
            );
        }
    }


    // Escape HTML used for temporary UI

    function escapeHtml(
        text
    ) {

        const div =
            document.createElement(
                "div"
            );

        div.textContent =
            text;

        return div.innerHTML;
    }


    // Upload zone click

    uploadZone.addEventListener(
        "click",
        () => {

            fileInput.click();

        }
    );


    // Drag over

    uploadZone.addEventListener(
        "dragover",
        event => {

            event.preventDefault();

            uploadZone.classList.add(
                "dragover"
            );
        }
    );


    // Drag leave

    uploadZone.addEventListener(
        "dragleave",
        () => {

            uploadZone.classList.remove(
                "dragover"
            );
        }
    );


    // Drop

    uploadZone.addEventListener(
        "drop",
        event => {

            event.preventDefault();

            uploadZone.classList.remove(
                "dragover"
            );

            handleFiles(
                event.dataTransfer.files
            );
        }
    );


    // File picker

    fileInput.addEventListener(
        "change",
        () => {

            handleFiles(
                fileInput.files
            );

            fileInput.value = "";
        }
    );


    // Send button

    sendBtn.addEventListener(
        "click",
        () => {

            sendMessage(
                userInput.value
            );
        }
    );


    // Enter key

    userInput.addEventListener(
        "keydown",
        event => {

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


    // Document delete buttons

    fileList.addEventListener(
        "click",
        event => {

            const deleteButton =
                event.target.closest(
                    ".delete-btn"
                );

            if (!deleteButton) {
                return;
            }

            if (
                deleteButton.disabled
            ) {
                return;
            }

            const documentId =
                deleteButton.dataset.documentId;

            if (!documentId) {
                return;
            }

            deleteDocument(
                documentId,
                deleteButton
            );
        }
    );


    // Copy bot response

    document.addEventListener(
        "click",
        event => {

            const copyButton =
                event.target.closest(
                    ".copy-btn"
                );

            if (!copyButton) {
                return;
            }

            const message =
                copyButton.closest(
                    ".message"
                );

            if (!message) {
                return;
            }

            const messageText =
                message.querySelector(
                    ".message-text"
                );

            if (!messageText) {
                return;
            }

            navigator.clipboard
                .writeText(
                    messageText.innerText
                )
                .then(() => {

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
                })
                .catch(
                    error => {

                        console.error(
                            "Copy failed:",
                            error
                        );
                    }
                );
        }
    );


    // Theme toggle

    themeToggle.addEventListener(
        "change",
        () => {

            document.body.classList.toggle(
                "light-theme"
            );

            if (
                themeToggle.checked
            ) {

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


    // Initial Eureka message

    addMessage(
        "bot",
        "Hello! I'm **Eureka** 💡. Upload a document or ask me anything."
    );


    // Load documents already uploaded

    loadDocuments();

});