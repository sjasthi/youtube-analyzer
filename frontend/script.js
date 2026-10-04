/*
    Frontend JavaScript responsibilities:
    1. Check that FastAPI is running.
    2. Read the user's channel ID and video limit.
    3. Call the FastAPI analytics endpoint.
    4. Put returned values into the page.
    5. Build the video table and posting-trend chart.
*/

// Get references to the HTML elements we need to update.
const backendStatus = document.getElementById("backendStatus");
const analyzeButton = document.getElementById("analyzeButton");
const channelInput = document.getElementById("channelInput");
const loadingMessage = document.getElementById("loadingMessage");
const errorMessage = document.getElementById("errorMessage");
const channelSection = document.getElementById("channelSection");
const analyticsSection = document.getElementById("analyticsSection");
const videosSection = document.getElementById("videosSection");
const channelThumbnail = document.getElementById("channelThumbnail");
const channelTitle = document.getElementById("channelTitle");
const channelHandle = document.getElementById("channelHandle");
const subscriberCount = document.getElementById("subscriberCount");
const videoCount = document.getElementById("videoCount");
const viewCount = document.getElementById("viewCount");
const retrievedCount = document.getElementById("retrievedCount");
const averageViews = document.getElementById("averageViews");
const averageLikes = document.getElementById("averageLikes");
const averageComments = document.getElementById("averageComments");
const averageDuration = document.getElementById("averageDuration");
const sampleTotalViews = document.getElementById("sampleTotalViews");
const engagementRate = document.getElementById("engagementRate");
const averageDaysBetweenUploads = document.getElementById("averageDaysBetweenUploads");
const uploadsPerMonth = document.getElementById("uploadsPerMonth");
const mostViewedTitle = document.getElementById("mostViewedTitle");
const mostViewedValue = document.getElementById("mostViewedValue");
const mostLikedTitle = document.getElementById("mostLikedTitle");
const mostLikedValue = document.getElementById("mostLikedValue");
const videoTableBody = document.getElementById("videoTableBody");

// Gets references to the RAG question and answer elements
const questionSection = document.getElementById("questionSection");
const questionInput = document.getElementById("questionInput");
const askQuestionButton = document.getElementById("askQuestionButton");
const answerText = document.getElementById("answerText");
const clearQuestionButton = document.getElementById("clearQuestionButton");
const exampleQuestions = document.querySelectorAll(".example-question");

// Default messages for the question and answer section
const defaultAnswer = "The answer will appear here.";

// Store the current Chart.js object so it can be replaced on the next search.
let uploadTrendChart = null;

// Store the current DataTables object so it can be replace on the next search.
let videoDataTable = null;

async function checkBackendConnection() {
    try {
        // fetch() sends an HTTP GET request to FastAPI.
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("Backend returned an error.");

        // Convert JSON into a normal JavaScript object.
        const data = await response.json();
        backendStatus.textContent = data.message;
        backendStatus.className = "success";
    } catch (error) {
        backendStatus.textContent = "Could not connect to the FastAPI backend.";
        backendStatus.className = "error";
    }
}


function formatNumber(value) {
    // Example: 1250000 -> 1,250,000
    return Number(value || 0).toLocaleString();
}


function formatDuration(totalSeconds) {
    // Convert seconds into MM:SS or H:MM:SS.
    const seconds = Math.round(Number(totalSeconds || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remaining = seconds % 60;

    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
    }

    return `${minutes}:${String(remaining).padStart(2, "0")}`;
}


function formatDate(dateString) {
    if (!dateString) return "Unknown";
    return new Date(dateString).toLocaleDateString();
}


function displayChannel(channel, retrievedVideoCount) {
    // Fill in channel summary values returned by Python.
    channelTitle.textContent = channel.title || "Unknown Channel";
    channelHandle.textContent = channel.custom_url || channel.channel_id;

    if (channel.thumbnail_url) {
        channelThumbnail.src = channel.thumbnail_url;
        channelThumbnail.classList.remove("hidden");
    } else {
        channelThumbnail.classList.add("hidden");
    }

    subscriberCount.textContent = formatNumber(channel.subscriber_count);
    videoCount.textContent = formatNumber(channel.video_count);
    viewCount.textContent = formatNumber(channel.view_count);
    retrievedCount.textContent = formatNumber(retrievedVideoCount);

    channelSection.classList.remove("hidden");
}


function displayAnalytics(analytics) {
    // These values were calculated by analytics_service.py, not JavaScript.
    averageViews.textContent = formatNumber(Math.round(analytics.average_views));
    averageLikes.textContent = formatNumber(Math.round(analytics.average_likes));
    averageComments.textContent = formatNumber(Math.round(analytics.average_comments));
    averageDuration.textContent = formatDuration(analytics.average_duration_seconds);
    sampleTotalViews.textContent = formatNumber(analytics.total_views);
    engagementRate.textContent = `${analytics.engagement_rate_percent}%`;

    // null means Python did not have enough dates to calculate the value.
    averageDaysBetweenUploads.textContent =
        analytics.average_days_between_uploads !== null
            ? analytics.average_days_between_uploads
            : "N/A";

    uploadsPerMonth.textContent =
        analytics.estimated_uploads_per_month !== null
            ? analytics.estimated_uploads_per_month
            : "N/A";

    if (analytics.most_viewed_video) {
        mostViewedTitle.textContent = analytics.most_viewed_video.title;
        mostViewedValue.textContent = `${formatNumber(analytics.most_viewed_video.view_count)} views`;
    }

    if (analytics.most_liked_video) {
        mostLikedTitle.textContent = analytics.most_liked_video.title;
        mostLikedValue.textContent = `${formatNumber(analytics.most_liked_video.like_count)} likes`;
    }

    displayUploadChart(analytics.uploads_by_month);
    analyticsSection.classList.remove("hidden");
}


function displayUploadChart(monthData) {
    // Separate monthly objects into labels and values for Chart.js.
    const labels = monthData.map(item => item.month);
    const counts = monthData.map(item => item.video_count);

    // Remove an old chart before creating a new one.
    if (uploadTrendChart) uploadTrendChart.destroy();

    const canvas = document.getElementById("uploadTrendChart");

    uploadTrendChart = new Chart(canvas, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Videos Published",
                data: counts,
            }],
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 },
                },
            },
        },
    });
}


function displayVideos(videos) {
    // Clear rows from the previous channel analysis.
    videoTableBody.innerHTML = "";

    if (videos.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = "No public videos were returned.";
        row.appendChild(cell);
        videoTableBody.appendChild(row);
    } else {
        // Create one table row for each returned video.
        videos.forEach(video => {
            const row = document.createElement("tr");

            const titleCell = document.createElement("td");
            titleCell.className = "video-cell";

            const title = document.createElement("span");
            title.className = "video-title";
            title.textContent = video.title || "Untitled";

            const id = document.createElement("span");
            id.className = "video-id";
            id.textContent = video.video_id;

            titleCell.appendChild(title);
            titleCell.appendChild(id);

            const publishedCell = document.createElement("td");
            publishedCell.textContent = formatDate(video.published_at);

            const durationCell = document.createElement("td");
            durationCell.textContent = formatDuration(video.duration_seconds);

            const viewsCell = document.createElement("td");
            viewsCell.textContent = formatNumber(video.view_count);

            const likesCell = document.createElement("td");
            likesCell.textContent = formatNumber(video.like_count);

            const commentsCell = document.createElement("td");
            commentsCell.textContent = formatNumber(video.comment_count);

            row.append(titleCell, publishedCell, durationCell, viewsCell, likesCell, commentsCell);
            videoTableBody.appendChild(row);
        });
    }

    // If DataTables was already created from a previous search, destroy it before creating a new one.
    if (videoDataTable) {
        videoDataTable.destroy();
    }

    // Create the new DataTable
    videoDataTable = $("#videoTable").DataTable({
        pageLength: 10,
        lengthMenu: [10, 25, 50],
        order: [[1, "desc"]],

        // Preevent the second header row from becoming another sortable heading row.
        orderCellsTop: true
    });

    // Connect the individual column filters
    $("#videoTable thead tr.filter-row th").each(function (columnIndex) {
        $("input", this).on("keyup change", function() {
            videoDataTable.column(columnIndex).search(this.value).draw();
        });
    });

    // Show the video section
    videosSection.classList.remove("hidden");
}


async function analyzeChannel() {
    // Read what the user typed/selected.
    const channelId = channelInput.value.trim();

    // Hide previous results while the new request is running.
    errorMessage.classList.add("hidden");
    channelSection.classList.add("hidden");
    analyticsSection.classList.add("hidden");
    videosSection.classList.add("hidden");
    questionSection.classList.add("hidden");

    if (channelId === "") {
        errorMessage.textContent = "Please enter a YouTube channel ID.";
        errorMessage.classList.remove("hidden");
        return;
    }

    loadingMessage.classList.remove("hidden");
    analyzeButton.disabled = true;

    try {
        // No video limit is sent. The backend follows YouTube pagination
        // until it reaches the end of the channel's uploads playlist.
        const url = `/api/channel/${encodeURIComponent(channelId)}/analytics`;
        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to analyze channel.");
        }

        // One backend request gives us channel, analytics, and videos.
        displayChannel(data.channel, data.retrieved_video_count);
        displayAnalytics(data.analytics);
        displayVideos(data.videos);

        // Shows the Q&A section after channel analysis
        questionSection.classList.remove("hidden");

    } catch (error) {
        errorMessage.textContent = `Error: ${error.message}`;
        errorMessage.classList.remove("hidden");
    } finally {
        // This always runs whether the request succeeds or fails.
        loadingMessage.classList.add("hidden");
        analyzeButton.disabled = false;
    }
}

// Handles the Ask Question button
async function askQuestion() {
    const question = questionInput.value.trim();

    // Checks for an empty question
    if (question === "") {
        answerText.textContent = "Please enter a question.";
        return;
    }
    
    answerText.textContent = "Loading answer...";
    
    try {
        const response = await fetch("/api/question", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            
            body: JSON.stringify({
                question: question,
            }),
        });
        
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to get an answer.");
        }

        answerText.textContent = data.answer;

    } catch (error) {
        answerText.textContent = `Error: ${error.message}`;
    }
}

// Asks the question when the button is clicked
askQuestionButton.addEventListener("click", askQuestion);

// Asks the question when Enter is pressed
questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter") askQuestion();
});

// Clears the question and answer
clearQuestionButton.addEventListener("click", () => {
    questionInput.value = "";
    answerText.textContent = defaultAnswer;
    askQuestionButton.disabled = true;
});

// Enables the Ask Question button when the user enters text
questionInput.addEventListener("input", () => {
    askQuestionButton.disabled = questionInput.value.trim() === "";
});

// Puts an example question into the question input
exampleQuestions.forEach(button => {
    button.addEventListener("click", () => {
        questionInput.value = button.textContent.trim();
        askQuestionButton.disabled = false;
    });
});

// Clicking the button starts the analysis.
analyzeButton.addEventListener("click", analyzeChannel);

// Pressing Enter in the channel input does the same thing.
channelInput.addEventListener("keydown", event => {
    if (event.key === "Enter") analyzeChannel();
});

// Clears the question and answer fields when the page loads
questionInput.value = "";
answerText.textContent = defaultAnswer;
askQuestionButton.disabled = true;

// Check FastAPI as soon as the page loads.
checkBackendConnection();
