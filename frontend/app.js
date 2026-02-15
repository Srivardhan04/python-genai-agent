const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('dropzone-file');
const fileSelectedLabel = document.getElementById('file-selected');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const resultsContainer = document.getElementById('results-container');
const noResultsPlaceholder = document.getElementById('no-results');
const apiStatusIndicator = document.getElementById('api-status');
const statDocs = document.getElementById('stat-docs');
const statChunks = document.getElementById('stat-chunks');

// State
let jobs = {};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    updateStats();
    // Refresh stats every 30 seconds
    setInterval(updateStats, 30000);
});

// Event Listeners
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileSelectedLabel.textContent = `Selected: ${e.target.files[0].name}`;
        fileSelectedLabel.classList.remove('hidden');
    }
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    setLoading(uploadBtn, true, 'Indexing...');
    showStatus(uploadStatus, 'Uploading and indexing document...', 'info');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload-document`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showStatus(uploadStatus, `Success! ${data.chunks_created} chunks indexed.`, 'success');
            updateStats();
            fileInput.value = '';
            fileSelectedLabel.classList.add('hidden');
        } else {
            showStatus(uploadStatus, `Error: ${data.detail || 'Upload failed'}`, 'error');
        }
    } catch (error) {
        showStatus(uploadStatus, 'Error connecting to backend API.', 'error');
    } finally {
        setLoading(uploadBtn, false, 'Index Document');
    }
});

questionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    setLoading(askBtn, true, 'Submitting...');

    try {
        const response = await fetch(`${API_BASE_URL}/ask-question`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok) {
            addJobToUI(data.job_id, question);
            pollJobStatus(data.job_id);
            questionInput.value = '';
            noResultsPlaceholder.classList.add('hidden');
        } else {
            alert(`Error: ${data.detail || 'Failed to submit question'}`);
        }
    } catch (error) {
        alert('Error connecting to backend API.');
    } finally {
        setLoading(askBtn, false, 'Ask Agents');
    }
});

// Helper Functions
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const statusSpan = apiStatusIndicator.querySelector('span:last-child');
        const statusDot = apiStatusIndicator.querySelector('span:first-child');
        
        if (response.ok) {
            statusSpan.textContent = 'API Online';
            statusDot.className = 'h-2 w-2 rounded-full bg-green-500 mr-2';
        } else {
            statusSpan.textContent = 'API Error';
            statusDot.className = 'h-2 w-2 rounded-full bg-red-500 mr-2';
        }
    } catch (error) {
        const statusSpan = apiStatusIndicator.querySelector('span:last-child');
        const statusDot = apiStatusIndicator.querySelector('span:first-child');
        statusSpan.textContent = 'API Offline';
        statusDot.className = 'h-2 w-2 rounded-full bg-red-500 mr-2';
    }
}

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (response.ok) {
            const data = await response.json();
            statDocs.textContent = data.total_documents;
            statChunks.textContent = data.total_chunks;
        }
    } catch (error) {
        console.error('Failed to update stats');
    }
}

function addJobToUI(jobId, question) {
    const template = document.getElementById('response-template');
    const clone = template.content.cloneNode(true);
    const container = clone.querySelector('div');
    
    container.id = `job-${jobId}`;
    container.querySelector('.response-question').textContent = question;
    container.querySelector('.response-time').textContent = new Date().toLocaleTimeString();
    
    const statusLabel = container.querySelector('.response-status');
    statusLabel.textContent = 'PENDING';
    statusLabel.classList.add('status-pending');
    
    resultsContainer.prepend(container);
    jobs[jobId] = container;
}

async function pollJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/job-status/${jobId}`);
            if (!response.ok) {
                clearInterval(pollInterval);
                return;
            }

            const data = await response.json();
            updateJobUI(jobId, data);

            if (data.status === 'COMPLETED' || data.status === 'FAILED') {
                clearInterval(pollInterval);
            }
        } catch (error) {
            console.error('Polling error', error);
            clearInterval(pollInterval);
        }
    }, 2000);
}

function updateJobUI(jobId, data) {
    const container = document.getElementById(`job-${jobId}`);
    if (!container) return;

    const statusLabel = container.querySelector('.response-status');
    const answerContainer = container.querySelector('.response-answer');
    const sourcesContainer = container.querySelector('.response-sources');
    const sourceTags = container.querySelector('.response-source-tags');

    // Update status
    statusLabel.textContent = data.status;
    statusLabel.className = 'response-status text-xs font-medium px-2 py-1 rounded-full';
    statusLabel.classList.add(`status-${data.status.toLowerCase()}`);

    if (data.status === 'COMPLETED') {
        answerContainer.textContent = data.result.answer;
        answerContainer.classList.remove('italic', 'text-gray-400');
        
        if (data.result.sources_used > 0) {
            sourcesContainer.classList.remove('hidden');
            sourceTags.innerHTML = '';
            for (let i = 0; i < data.result.sources_used; i++) {
                const tag = document.createElement('span');
                tag.className = 'bg-gray-100 text-gray-600 px-2 py-1 rounded text-[10px] font-medium';
                tag.textContent = `Source Chunk ${i+1}`;
                sourceTags.appendChild(tag);
            }
        }
    } else if (data.status === 'FAILED') {
        answerContainer.textContent = `Execution failed: ${data.error || 'Unknown error'}`;
        answerContainer.classList.add('text-red-500');
    }
}

function setLoading(button, isLoading, text) {
    button.disabled = isLoading;
    button.textContent = text;
}

function showStatus(element, message, type) {
    element.textContent = message;
    element.classList.remove('hidden', 'text-blue-600', 'text-green-600', 'text-red-600');
    
    const colorClass = {
        'info': 'text-blue-600',
        'success': 'text-green-600',
        'error': 'text-red-600'
    }[type];
    
    element.classList.add(colorClass);
    element.classList.remove('hidden');
}
