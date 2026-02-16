class ChatApp {
    constructor() {
        this.chatBox = document.getElementById('chat-box');
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.uploadForm = document.getElementById('upload-form');
        this.dragDropArea = document.getElementById('drag-drop-area');
        this.fileInput = document.getElementById('file-input');
        this.fileMessage = document.getElementById('file-selected-message');
        this.voiceButton = document.getElementById('voice-button');
        this.recordingIndicator = document.getElementById('recording-indicator');
        
        this.isRecording = false;
        this.recordingTimeout = null;
        this.conversationHistory = [];
        this.selectedLanguage = 'en';
        
        this.initializeVoiceInput();
        this.initializeEventListeners();
    }

    initializeVoiceInput() {
        this.voiceButton.addEventListener('click', () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.shiftKey && event.key === 'R') {
                if (this.isRecording) {
                    this.stopRecording();
                } else {
                    this.startRecording();
                }
            }
        });
    }

    async startRecording() {
        if (this.isRecording) return;

        try {
            const response = await fetch('/start_recording', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: this.selectedLanguage })
            });

            if (response.ok) {
                this.isRecording = true;
                this.voiceButton.classList.add('recording');
                this.recordingIndicator.classList.remove('hidden');
                this.voiceButton.querySelector('i').classList.remove('fa-microphone');
                this.voiceButton.querySelector('i').classList.add('fa-microphone-alt');
                
                this.recordingTimeout = setTimeout(() => {
                    if (this.isRecording) this.stopRecording();
                }, 10000);
            }
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not start recording');
        }
    }

    async stopRecording() {
        if (!this.isRecording) return;
        
        clearTimeout(this.recordingTimeout);
        
        try {
            const response = await fetch('/stop_recording', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: this.selectedLanguage })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.text) {
                    this.chatInput.value = data.text;
                    this.handleSendMessage();
                }
            }
        } catch (error) {
            console.error('Error stopping recording:', error);
            this.showError('Could not process speech');
        } finally {
            this.isRecording = false;
            this.voiceButton.classList.remove('recording');
            this.recordingIndicator.classList.add('hidden');
            this.voiceButton.querySelector('i').classList.remove('fa-microphone-alt');
            this.voiceButton.querySelector('i').classList.add('fa-microphone');
        }
    }

    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.handleSendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSendMessage();
        });
        this.uploadForm.addEventListener('submit', (e) => this.handleFileUpload(e));
        this.initializeDragDrop();
        this.fileInput.addEventListener('change', () => this.updateFileSelection());
    }

    async handleSendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        this.chatInput.value = '';
        const loadingIndicator = this.showLoadingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message,
                    conversation_history: this.conversationHistory,
                    target_lang: this.selectedLanguage 
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            
            this.conversationHistory = data.conversation_history;
            this.addMessage('bot', data.response);
        } catch (error) {
            this.addMessage('error', 'Sorry, something went wrong. Please try again.');
            console.error('Error:', error);
        } finally {
            loadingIndicator.remove();
        }
    }

    async handleFileUpload(event) {
        event.preventDefault();
        if (!this.fileInput.files[0]) return;

        const formData = new FormData();
        formData.append('file', this.fileInput.files[0]);
        
        const progressBar = this.showUploadProgress();

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) throw new Error(data.error || 'Upload failed');
            
            this.addMessage('system', 'File uploaded successfully! You can now ask questions about it.');
            this.fileInput.value = '';
            this.fileMessage.textContent = 'No file chosen';
        } catch (error) {
            this.addMessage('error', `Upload failed: ${error.message}`);
        } finally {
            progressBar.remove();
        }
    }

    initializeDragDrop() {
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dragDropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                this.dragDropArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dragDropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                this.dragDropArea.classList.remove('dragover');
                if (eventName === 'drop') {
                    this.fileInput.files = e.dataTransfer.files;
                    this.updateFileSelection();
                }
            });
        });
    }

    updateFileSelection() {
        const file = this.fileInput.files[0];
        this.fileMessage.textContent = file ? `Selected: ${file.name}` : 'No file chosen';
    }

    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const avatar = document.createElement('img');
        avatar.className = 'avatar';
        avatar.src = `/static/images/${type}-avatar.png`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = content;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(textDiv);
        this.chatBox.appendChild(messageDiv);
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
    }

    showLoadingIndicator() {
        const loading = document.createElement('div');
        loading.className = 'message bot loading';
        loading.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        this.chatBox.appendChild(loading);
        return loading;
    }

    showUploadProgress() {
        const progress = document.createElement('div');
        progress.className = 'upload-progress';
        progress.innerHTML = '<div class="progress-bar"></div>';
        this.uploadForm.appendChild(progress);
        return progress;
    }

    showError(message) {
        const errorToast = document.createElement('div');
        errorToast.className = 'error-toast';
        errorToast.textContent = message;
        document.body.appendChild(errorToast);
        setTimeout(() => errorToast.remove(), 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});