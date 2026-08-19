document.addEventListener('DOMContentLoaded', () => {

    document.body.classList.remove('loading');

    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');

    const documentCount = document.getElementById('document-count');
    const clearDocumentsBtn = document.getElementById('clear-documents-btn');

    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    const chatStatus = document.getElementById('chat-status');

    const ragStatus = document.getElementById('rag-status');
    const ragStatusText = document.getElementById('rag-status-text');


    const MAX_DOCUMENTS = 5;
    const MAX_FILE_SIZE = 25 * 1024 * 1024;

    const allowedExtensions = [
        '.pdf',
        '.docx'
    ];


    // Update the small status shown in the chat header.
    function setChatStatus(text, busy = false) {

        chatStatus.classList.toggle(
            'busy',
            busy
        );

        const dot = chatStatus.querySelector(
            '.status-dot'
        );

        if (dot) {
            dot.style.backgroundColor = busy
                ? 'var(--warning-color)'
                : 'var(--success-color)';
        }

        chatStatus.lastChild.textContent = ` ${text}`;
    }


    // Show or hide the RAG status message.
    function setRagStatus(visible, text = '') {

        ragStatus.classList.toggle(
            'hidden',
            !visible
        );

        if (text) {
            ragStatusText.textContent = text;
        }
    }


    // Create a chat message element.
    function createMessageDiv(sender) {

        const messageDiv = document.createElement('div');

        messageDiv.className = `message ${sender}`;

        const textElement = document.createElement('div');

        textElement.className = 'message-text';

        messageDiv.appendChild(textElement);

        return messageDiv;
    }


    // Add a normal message to the chat.
    function addMessage(sender, text) {

        const messageDiv = createMessageDiv(sender);

        const textElement = messageDiv.querySelector(
            '.message-text'
        );

        if (sender === 'bot') {
            textElement.innerHTML = marked.parse(text);
        } else {
            textElement.textContent = text;
        }

        if (sender === 'bot') {

            const copyBtn = document.createElement('i');

            copyBtn.className =
                'fas fa-copy copy-btn';

            copyBtn.title = 'Copy text';

            messageDiv.appendChild(copyBtn);
        }

        chatBox.appendChild(messageDiv);

        chatBox.scrollTop =
            chatBox.scrollHeight;
    }


    // Format the source section returned by the backend.
    function formatBotResponse(text) {

        const sourceMarker = '\n\nSources:\n';

        if (!text.includes(sourceMarker)) {
            return marked.parse(text);
        }

        const parts = text.split(
            sourceMarker
        );

        const answer = parts[0];

        const sourceText = parts[1] || '';

        const sources = sourceText
            .split('\n')
            .map(source => source.trim())
            .filter(source => source.startsWith('•'));

        let html = marked.parse(answer);

        if (sources.length) {

            html += `
                <div class="sources">
                    <div class="sources-title">
                        <i class="fas fa-book-open"></i>
                        Sources
                    </div>
            `;

            sources.forEach(source => {

                const cleanSource =
                    source.replace(/^•\s*/, '');

                html += `
                    <div class="source-item">
                        <i class="fas fa-file-lines"></i>
                        ${escapeHtml(cleanSource)}
                    </div>
                `;
            });

            html += '</div>';
        }

        return html;
    }


    // Keep source filenames safe when adding them to the page.
    function escapeHtml(text) {

        const div =
            document.createElement('div');

        div.textContent = text;

        return div.innerHTML;
    }


    // Send a message to the backend.
    async function sendMessage(message) {

        if (!message.trim()) {
            return;
        }

        addMessage(
            'user',
            message
        );

        userInput.value = '';

        sendBtn.disabled = true;

        setChatStatus(
            'Thinking...',
            true
        );

        const botMessageDiv =
            createMessageDiv('bot');

        const textElement =
            botMessageDiv.querySelector(
                '.message-text'
            );

        chatBox.appendChild(
            botMessageDiv
        );

        chatBox.scrollTop =
            chatBox.scrollHeight;

        try {

            const response = await fetch(
                '/chat',
                {
                    method: 'POST',

                    headers: {
                        'Content-Type':
                            'application/json'
                    },

                    body: JSON.stringify({
                        message
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
                    'No response body.'
                );
            }


            const reader =
                response.body.getReader();

            const decoder =
                new TextDecoder();

            let fullText = '';


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
                    formatBotResponse(
                        fullText + '▋'
                    );


                const isScrolledToBottom =
                    chatBox.scrollHeight -
                    chatBox.clientHeight <=
                    chatBox.scrollTop + 40;


                if (isScrolledToBottom) {

                    chatBox.scrollTop =
                        chatBox.scrollHeight;
                }
            }


            textElement.innerHTML =
                formatBotResponse(
                    fullText
                );


        } catch (error) {

            console.error(
                'Chat error:',
                error
            );

            textElement.innerHTML =
                marked.parse(
                    '❌ I could not connect to the server. Please try again.'
                );

        } finally {

            sendBtn.disabled = false;

            setChatStatus(
                'Ready',
                false
            );

            userInput.focus();
        }
    }


    // Display one uploaded document in the sidebar.
    function displayFile(
        filename,
        status = 'Ready',
        statusClass = 'ready'
    ) {

        const existing =
            [...fileList.querySelectorAll(
                '.file-item'
            )].find(
                item =>
                    item.dataset.filename === filename
            );


        if (existing) {

            const statusElement =
                existing.querySelector(
                    '.file-status'
                );

            statusElement.textContent =
                status;

            statusElement.className =
                `file-status ${statusClass}`;

            return existing;
        }


        const noFiles =
            fileList.querySelector(
                '.no-files'
            );


        if (noFiles) {
            noFiles.style.display = 'none';
        }


        const fileItem =
            document.createElement('div');


        fileItem.className =
            'file-item';


        fileItem.dataset.filename =
            filename;


        fileItem.innerHTML = `
            <div class="file-item-info">
                <i class="fas fa-file-lines"></i>

                <span
                    class="file-name"
                    title="${escapeHtml(filename)}"
                >
                    ${escapeHtml(filename)}
                </span>
            </div>

            <span class="file-status ${statusClass}">
                ${status}
            </span>
        `;


        fileList.appendChild(
            fileItem
        );


        return fileItem;
    }


    // Update the document counter.
    function updateDocumentCount(count) {

        documentCount.textContent =
            `${count} / ${MAX_DOCUMENTS}`;


        clearDocumentsBtn.disabled =
            count === 0;
    }


    // Refresh the sidebar from the backend.
    async function loadDocuments() {

        try {

            const response =
                await fetch(
                    '/documents'
                );


            if (!response.ok) {
                return;
            }


            const data =
                await response.json();


            fileList.innerHTML = '';


            if (
                !data.documents ||
                data.documents.length === 0
            ) {

                fileList.innerHTML = `
                    <p class="no-files">
                        📪 No documents uploaded yet.
                    </p>
                `;

            } else {

                data.documents.forEach(
                    filename => {

                        displayFile(
                            filename,
                            'Ready',
                            'ready'
                        );
                    }
                );
            }


            updateDocumentCount(
                data.count || 0
            );


        } catch (error) {

            console.error(
                'Could not load documents:',
                error
            );
        }
    }


    // Check a selected file before uploading it.
    function validateFile(file) {

        const extension =
            '.' +
            file.name
                .split('.')
                .pop()
                .toLowerCase();


        if (!allowedExtensions.includes(
            extension
        )) {

            alert(
                'Unsupported file type. Please upload a PDF or DOCX file.'
            );

            return false;
        }


        if (file.size > MAX_FILE_SIZE) {

            alert(
                `${file.name} is larger than 25 MB. Please choose a smaller file.`
            );

            return false;
        }


        return true;
    }


    // Upload one file to the backend.
    async function uploadFile(file) {

        const fileItem =
            displayFile(
                file.name,
                'Uploading...',
                'processing'
            );


        try {

            const formData =
                new FormData();


            formData.append(
                'file',
                file
            );


            const response =
                await fetch(
                    '/upload',
                    {
                        method: 'POST',
                        body: formData
                    }
                );


            let data = {};

            try {
                data =
                    await response.json();
            } catch {
                data = {};
            }


            if (
                response.ok &&
                data.success
            ) {

                const statusElement =
                    fileItem.querySelector(
                        '.file-status'
                    );


                statusElement.textContent =
                    'Ready ✓';


                statusElement.className =
                    'file-status ready';


                updateDocumentCount(
                    data.document_count
                );


                setChatStatus(
                    'Documents ready',
                    false
                );


            } else {

                throw new Error(
                    data.error ||
                    'Upload failed.'
                );
            }


        } catch (error) {

            console.error(
                'Upload error:',
                error
            );


            fileItem.remove();


            const remainingFiles =
                fileList.querySelectorAll(
                    '.file-item'
                ).length;


            if (remainingFiles === 0) {

                fileList.innerHTML = `
                    <p class="no-files">
                        📪 No documents uploaded yet.
                    </p>
                `;
            }


            alert(
                `${file.name}: ${error.message}`
            );


            await loadDocuments();
        }
    }


    // Handle selected or dropped files.
    async function handleFiles(files) {

        const selectedFiles =
            Array.from(files);


        if (!selectedFiles.length) {
            return;
        }


        const currentCount =
            fileList.querySelectorAll(
                '.file-item'
            ).length;


        const availableSlots =
            MAX_DOCUMENTS -
            currentCount;


        if (availableSlots <= 0) {

            alert(
                `You can upload a maximum of ${MAX_DOCUMENTS} documents.`
            );

            return;
        }


        const filesToUpload =
            selectedFiles.slice(
                0,
                availableSlots
            );


        if (
            selectedFiles.length >
            availableSlots
        ) {

            alert(
                `Only ${availableSlots} more document(s) can be uploaded.`
            );
        }


        for (const file of filesToUpload) {

            if (!validateFile(file)) {
                continue;
            }


            const alreadyUploaded =
                [...fileList.querySelectorAll(
                    '.file-item'
                )].some(
                    item =>
                        item.dataset.filename ===
                        file.name
                );


            if (alreadyUploaded) {

                alert(
                    `${file.name} is already uploaded.`
                );

                continue;
            }


            await uploadFile(file);
        }


        fileInput.value = '';

        await loadDocuments();
    }


    // Clear the complete document collection.
    async function clearDocuments() {

        const confirmed =
            confirm(
                'Remove all uploaded documents?'
            );


        if (!confirmed) {
            return;
        }


        clearDocumentsBtn.disabled =
            true;


        try {

            const response =
                await fetch(
                    '/clear_document',
                    {
                        method: 'POST'
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    'Could not clear documents.'
                );
            }


            fileList.innerHTML = `
                <p class="no-files">
                    📪 No documents uploaded yet.
                </p>
            `;


            updateDocumentCount(0);


            setRagStatus(
                false
            );


            setChatStatus(
                'Ready',
                false
            );


            addMessage(
                'bot',
                'Document context cleared. You can upload new documents or ask me anything.'
            );


        } catch (error) {

            console.error(
                'Clear documents error:',
                error
            );


            alert(
                error.message
            );


            await loadDocuments();
        }
    }


    // Switch between dark and light themes.
    themeToggle.addEventListener(
        'change',
        () => {

            document.body.classList.toggle(
                'light-theme'
            );


            if (themeToggle.checked) {

                themeIcon.className =
                    'fas fa-moon';

                themeText.textContent =
                    'Dark Mode';

            } else {

                themeIcon.className =
                    'fas fa-sun';

                themeText.textContent =
                    'Light Mode';
            }
        }
    );


    // Open the file browser.
    uploadZone.addEventListener(
        'click',
        () => {
            fileInput.click();
        }
    );


    // Highlight the upload area while dragging.
    uploadZone.addEventListener(
        'dragover',
        event => {

            event.preventDefault();

            uploadZone.classList.add(
                'dragover'
            );
        }
    );


    uploadZone.addEventListener(
        'dragleave',
        () => {

            uploadZone.classList.remove(
                'dragover'
            );
        }
    );


    // Handle dropped files.
    uploadZone.addEventListener(
        'drop',
        event => {

            event.preventDefault();

            uploadZone.classList.remove(
                'dragover'
            );

            handleFiles(
                event.dataTransfer.files
            );
        }
    );


    // Handle files selected through the browser.
    fileInput.addEventListener(
        'change',
        () => {

            handleFiles(
                fileInput.files
            );
        }
    );


    // Send a message when the send button is clicked.
    sendBtn.addEventListener(
        'click',
        () => {

            sendMessage(
                userInput.value
            );
        }
    );


    // Send a message when Enter is pressed.
    userInput.addEventListener(
        'keydown',
        event => {

            if (
                event.key === 'Enter' &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage(
                    userInput.value
                );
            }
        }
    );


    // Copy bot responses to the clipboard.
    document.addEventListener(
        'click',
        async event => {

            if (
                event.target.classList.contains(
                    'copy-btn'
                )
            ) {

                const messageText =
                    event.target
                        .closest('.message')
                        .querySelector(
                            '.message-text'
                        )
                        .innerText;


                try {

                    await navigator.clipboard.writeText(
                        messageText
                    );


                    event.target.classList.replace(
                        'fa-copy',
                        'fa-check'
                    );


                    setTimeout(
                        () => {

                            event.target.classList.replace(
                                'fa-check',
                                'fa-copy'
                            );

                        },
                        1500
                    );


                } catch (error) {

                    console.error(
                        'Copy failed:',
                        error
                    );
                }
            }
        }
    );


    // Show a welcome message.
    addMessage(
        'bot',
        "Hello! I'm **Eureka** 💡. Upload documents or ask me anything."
    );


    // Load any documents already stored in the current session.
    loadDocuments();

});