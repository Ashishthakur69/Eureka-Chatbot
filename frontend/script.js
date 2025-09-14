document.addEventListener('DOMContentLoaded', () => {
    // --- Remove the loading class to show the app ---
    document.body.classList.remove('loading');

    // --- Get references to all necessary HTML elements ---
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById('send-btn');
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    // --- ** NEW STREAMING sendMessage FUNCTION ** ---
    async function sendMessage(message) {
        if (!message.trim()) return;

        addMessage("user", message);
        userInput.value = "";
        removeTypingIndicator(); // We don't need the "..." typing indicator anymore

        // Create a new, empty message bubble for the bot's response
        const botMessageDiv = createMessageDiv("bot");
        const textElement = botMessageDiv.querySelector('.message-text');
        chatBox.appendChild(botMessageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            if (!response.body) {
                throw new Error("No response body.");
            }

            // Read the response as a stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                // Append the new chunk of text
                fullText += decoder.decode(value, { stream: true });
                // Update the message bubble with the Markdown-parsed text
                textElement.innerHTML = marked.parse(fullText + '▋'); // Add a cursor effect

                // Smart scroll
                const isScrolledToBottom = chatBox.scrollHeight - chatBox.clientHeight <= chatBox.scrollTop + 20;
                if (isScrolledToBottom) {
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            }
            // Final update to remove the cursor
            textElement.innerHTML = marked.parse(fullText);

        } catch (err) {
            textElement.innerHTML = marked.parse(`❌ Connection error: Could not reach the server. 😞`);
            console.error(err);
        }
    }
    
    // Helper function to create a message structure
    function createMessageDiv(sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `<div class="message-text"></div>`;
        return messageDiv;
    }


    // --- (The rest of the script remains largely the same) ---
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        const formattedText = sender === 'bot' ? marked.parse(text) : text;
        
        messageDiv.innerHTML = formattedText;
        if (sender === 'bot') {
            const copyBtn = document.createElement('i');
            copyBtn.className = 'fas fa-copy copy-btn';
            copyBtn.title = 'Copy text';
            messageDiv.appendChild(copyBtn);
        }
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    // We no longer need showTypingIndicator or removeTypingIndicator
    const showTypingIndicator = () => {};
    const removeTypingIndicator = () => {};

    // All other functions and event listeners
    themeToggle.addEventListener('change', () => {
        document.body.classList.toggle('light-theme');

        if (themeToggle.checked) {
            // If the checkbox is checked, it's Light Mode
            themeIcon.className = 'fas fa-moon';
            themeText.textContent = 'Dark Mode ';
        } else {
            // If it's not checked, it's Dark Mode
            themeIcon.className = 'fas fa-sun';
            themeText.textContent = 'Light Mode ';
        }
    });
    function handleFiles(files) {
        if (fileList.querySelector('.file-item')) { alert("Please remove the current document before uploading a new one. ☝️"); return; }
        if (files.length > 1) { alert("You can only upload one document at a time. ☝️"); return; }
        const file = files[0];
        const allowedFileTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!allowedFileTypes.includes(file.type)) { alert(`Unsupported file type: ${file.name} 🚫. Please upload a PDF, TXT, or DOCX file. 📄`); return; }
        fileList.querySelector('.no-files').style.display = 'none';
        displayFile(file);
        uploadFile(file);
    }
    function displayFile(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `<span><i class="fas fa-file-alt"></i> ${file.name}</span><span class="file-status">Uploading ⏳...</span><i class="fas fa-trash delete-btn" title="Remove file"></i>`;
        fileList.appendChild(fileItem);
    }
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        const fileStatusSpan = fileList.querySelector('.file-item:last-child .file-status');
        fileStatusSpan.textContent = 'Processing... ⚙️';
        fileStatusSpan.classList.add('processing');
        try {
            const response = await fetch("/upload", { method: "POST", body: formData });
            const data = await response.json();
            if (response.ok && data.success) { 
                fileStatusSpan.textContent = 'Ready ✅'; 
                fileStatusSpan.classList.remove('processing');
            }
            else { throw new Error(data.error || 'Upload failed'); }
        } catch (error) {
            fileStatusSpan.textContent = 'Error ❌';
            fileStatusSpan.classList.remove('processing');
            console.error('Upload Error:', error);
        }
    }
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); });
    uploadZone.addEventListener('drop', (e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); });
    fileInput.addEventListener('change', () => handleFiles(fileInput.files));
    sendBtn.addEventListener('click', () => sendMessage(userInput.value));
    userInput.addEventListener('keyup', (e) => { if (e.key === 'Enter') sendMessage(userInput.value); });
    document.addEventListener('click', async (e) => {
        if (e.target.classList.contains('delete-btn')) {
            await fetch('/clear_document', { method: 'POST' });
            e.target.parentElement.remove();
            if (!fileList.querySelector('.file-item')) { fileList.querySelector('.no-files').style.display = 'block'; }
        }
        if (e.target.classList.contains('copy-btn')) {
            const messageText = e.target.parentElement.textContent;
            navigator.clipboard.writeText(messageText).then(() => {
                e.target.classList.replace('fa-copy', 'fa-check');
                setTimeout(() => e.target.classList.replace('fa-check', 'fa-copy'), 1500);
            });
        }
    });
    addMessage("bot", "Hello! I'm **Eureka** 💡. Upload a document or ask me anything.");
});