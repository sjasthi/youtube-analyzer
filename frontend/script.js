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
const clearInputButton = document.getElementById("clearInputButton");
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

// Dark-mode toggle elements.
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");

// Gets references to the RAG question and answer elements
const questionSection = document.getElementById("questionSection");
const questionInput = document.getElementById("questionInput");
const askQuestionButton = document.getElementById("askQuestionButton");
const indexRagButton = document.getElementById("indexRagButton");
const transcriptStatus = document.getElementById("transcriptStatus");
const clearQuestionButton = document.getElementById("clearQuestionButton");
const newChatButton = document.getElementById("newChatButton");
const exampleQuestions = document.querySelectorAll(".example-question");
const openChatButton = document.getElementById("openChatButton");
const closeChatButton = document.getElementById("closeChatButton");
const chatModal = document.getElementById("chatModal");
const chatMessages = document.getElementById("chatMessages");
const cancelIndexButton = document.getElementById("cancelIndexButton");
const indexProgressPanel = document.getElementById("indexProgressPanel");
const indexProgressText = document.getElementById("indexProgressText");
const indexProgressCount = document.getElementById("indexProgressCount");
const indexProgressBar = document.getElementById("indexProgressBar");

// FP9 channel-wide summary elements.
const channelSummarySection = document.getElementById("channelSummarySection");
const generateChannelSummaryButton = document.getElementById("generateChannelSummaryButton");
const copyChannelSummaryButton = document.getElementById("copyChannelSummaryButton");
const channelSummaryStatus = document.getElementById("channelSummaryStatus");
const channelSummaryResult = document.getElementById("channelSummaryResult");
const channelSummaryText = document.getElementById("channelSummaryText");
const channelSummarySourceList = document.getElementById("channelSummarySourceList");

// Transcript modal elements
const copyTranscriptModalButton = document.getElementById("copyTranscriptModalButton");
const transcriptModalText = document.getElementById("transcriptModalText");

// Store the current Chart.js object so it can be replaced on the next search.
let uploadTrendChart = null;

// Store the current DataTables object so it can be replace on the next search.
let videoDataTable = null;

// Resolved UC channel ID used by the RAG question endpoint.
let activeChannelId = null;
let activeIndexJobId = null;
let indexJobPoller = null;
let videoStatusPoller = null;
let currentVideos = [];
let chatSessionId = sessionStorage.getItem("ragChatSessionId") || (crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random()}`);
sessionStorage.setItem("ragChatSessionId", chatSessionId);

// Returns true when the saved/active page theme is dark.
function isDarkMode() {
    return document.documentElement.classList.contains("dark-mode");
}


// Keep the button text, icon, and accessibility state in sync with the theme.
function updateThemeToggle() {
    const darkModeEnabled = isDarkMode();

    themeIcon.textContent = darkModeEnabled ? "☀️" : "🌙";
    themeText.textContent = darkModeEnabled ? "Light Mode" : "Dark Mode";
    themeToggle.setAttribute("aria-pressed", String(darkModeEnabled));
    themeToggle.setAttribute(
        "aria-label",
        darkModeEnabled ? "Switch to light mode" : "Switch to dark mode"
    );
}


// Change Chart.js text/grid colors when the page theme changes.
function applyChartTheme() {
    if (!uploadTrendChart) return;

    const styles = getComputedStyle(document.documentElement);
    const textColor = styles.getPropertyValue("--text").trim();
    const gridColor = styles.getPropertyValue("--chart-grid").trim();

    uploadTrendChart.options.plugins.legend.labels.color = textColor;
    uploadTrendChart.options.scales.x.ticks.color = textColor;
    uploadTrendChart.options.scales.y.ticks.color = textColor;
    uploadTrendChart.options.scales.x.grid.color = gridColor;
    uploadTrendChart.options.scales.y.grid.color = gridColor;
    uploadTrendChart.update();
}


// Switch themes and remember the user's choice in this browser.
function toggleTheme() {
    const darkModeEnabled = document.documentElement.classList.toggle("dark-mode");

    localStorage.setItem("theme", darkModeEnabled ? "dark" : "light");
    updateThemeToggle();
    applyChartTheme();
}


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
        const video = analytics.most_viewed_video;
        const videoLink = document.createElement("a");
        const thumbnail = document.createElement("img");

        videoLink.href = `https://www.youtube.com/watch?v=${encodeURIComponent(video.video_id)}`;
        videoLink.target = "_blank";
        videoLink.rel = "noopener noreferrer";
        videoLink.className = "video-title";
        videoLink.textContent = video.title;

        thumbnail.src = `https://img.youtube.com/vi/${encodeURIComponent(video.video_id)}/mqdefault.jpg`;
        thumbnail.alt = `Thumbnail for ${video.title}`;
        thumbnail.className = "analytics-video-thumbnail";

        mostViewedTitle.replaceChildren(thumbnail, videoLink);
        mostViewedValue.textContent = `${formatNumber(video.view_count)} views`;
    }

    if (analytics.most_liked_video) {
        const video = analytics.most_liked_video;
        const videoLink = document.createElement("a");
        const thumbnail = document.createElement("img");

        videoLink.href = `https://www.youtube.com/watch?v=${encodeURIComponent(video.video_id)}`;
        videoLink.target = "_blank";
        videoLink.rel = "noopener noreferrer";
        videoLink.className = "video-title";
        videoLink.textContent = video.title;

        thumbnail.src = `https://img.youtube.com/vi/${encodeURIComponent(video.video_id)}/mqdefault.jpg`;
        thumbnail.alt = `Thumbnail for ${video.title}`;
        thumbnail.className = "analytics-video-thumbnail";

        mostLikedTitle.replaceChildren(thumbnail, videoLink);
        mostLikedValue.textContent = `${formatNumber(video.like_count)} likes`;
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

    // Read the current CSS theme so Chart.js text/grid colors match the page.
    const styles = getComputedStyle(document.documentElement);
    const textColor = styles.getPropertyValue("--text").trim();
    const gridColor = styles.getPropertyValue("--chart-grid").trim();

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
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0,
                        color: textColor
                    },
                    grid: {
                        color: gridColor
                    }
                },
            },
        },
    });
}


function videoTypeLabel(video) {
    const seconds = Number(video.duration_seconds || 0);
    return seconds > 0 && seconds <= 180 ? "Short-form candidate" : "Regular / unknown";
}


function makeStatusBadge(kind, label, className, tooltip = "") {
    const badge = document.createElement("span");
    badge.className = `status-badge ${className}`;
    badge.dataset.kind = kind;
    badge.textContent = label;

    if (tooltip) {
        badge.title = tooltip;
    }

    return badge;
}


function displayVideos(videos) {
    currentVideos = videos || [];
    if (videoDataTable) {
        videoDataTable.destroy();
        videoDataTable = null;
    }
    videoTableBody.innerHTML = "";

    if (currentVideos.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 11;
        cell.textContent = "No public videos were returned.";
        row.appendChild(cell);
        videoTableBody.appendChild(row);
    } else {
        currentVideos.forEach(video => {
            const row = document.createElement("tr");
            row.dataset.videoId = video.video_id;
            const videoUrl = `https://www.youtube.com/watch?v=${video.video_id}`;

            const thumbnailCell = document.createElement("td");
            thumbnailCell.className = "video-thumbnail-cell";
            if (video.thumbnail_url) {
                const thumbnailLink = document.createElement("a");
                thumbnailLink.href = videoUrl;
                thumbnailLink.target = "_blank";
                thumbnailLink.rel = "noopener noreferrer";
                const thumbnail = document.createElement("img");
                thumbnail.className = "video-thumbnail";
                thumbnail.src = video.thumbnail_url;
                thumbnail.alt = `${video.title || "Video"} thumbnail`;
                thumbnail.loading = "lazy";
                thumbnailLink.appendChild(thumbnail);
                thumbnailCell.appendChild(thumbnailLink);
            } else {
                thumbnailCell.textContent = "No image";
            }

            const titleCell = document.createElement("td");
            titleCell.className = "video-cell";
            const title = document.createElement("a");
            title.className = "video-title";
            title.textContent = video.title || "Untitled";
            title.href = videoUrl;
            title.target = "_blank";
            title.rel = "noopener noreferrer";
            const id = document.createElement("span");
            id.className = "video-id";
            id.textContent = video.video_id;
            titleCell.append(title, id);

            // Copy Video Link button
            const copyLinkButton = document.createElement("button");
            copyLinkButton.type = "button";
            copyLinkButton.className = "copy-video-link-button";
            copyLinkButton.setAttribute("aria-label", `Copy link for ${video.title || "Untitled"}`);
            copyLinkButton.textContent = "Copy Link";
            
            copyLinkButton.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(videoUrl);
                    copyLinkButton.textContent = "Copied!";
                    
                    setTimeout(() => {
                        copyLinkButton.textContent = "Copy Link";
                    }, 2000);
                } catch (error) {
                    copyLinkButton.textContent = "Copy failed";
                }
            });
            
            titleCell.appendChild(copyLinkButton);

            const typeCell = document.createElement("td");
            typeCell.className = "video-type-cell compact-status-cell";
            const typeLabel = videoTypeLabel(video);
            const typeBadge = makeStatusBadge(
                "type",
                typeLabel === "Short-form candidate" ? "Short-form" : "Regular",
                typeLabel === "Short-form candidate" ? "status-info" : "status-neutral"
            );
            typeCell.appendChild(typeBadge);
            typeCell.title = typeLabel === "Short-form candidate"
                ? "Duration is 3 minutes or less. The YouTube Data API does not expose the aspect-ratio signal needed to prove this is a Short."
                : "Not classified as a short-form candidate from duration alone";

            const transcriptCell = document.createElement("td");
            transcriptCell.className = "transcript-status-cell compact-status-cell";
            transcriptCell.appendChild(makeStatusBadge("transcript", "Checking…", "status-pending"));

            const ragCell = document.createElement("td");
            ragCell.className = "rag-status-cell compact-status-cell";
            ragCell.appendChild(makeStatusBadge("rag", "Checking…", "status-pending"));

            const publishedCell = document.createElement("td");
            publishedCell.textContent = formatDate(video.published_at);
            publishedCell.dataset.order = new Date(video.published_at).getTime();

            const durationCell = document.createElement("td");
            durationCell.textContent = formatDuration(video.duration_seconds);
            durationCell.dataset.order = video.duration_seconds;

            const viewsCell = document.createElement("td");
            viewsCell.textContent = formatNumber(video.view_count);
            viewsCell.dataset.order = video.view_count;
            const likesCell = document.createElement("td");
            likesCell.textContent = formatNumber(video.like_count);
            likesCell.dataset.order = video.like_count;
            const commentsCell = document.createElement("td");
            commentsCell.textContent = formatNumber(video.comment_count);
            commentsCell.dataset.order = video.comment_count;
            const urlCell = document.createElement("td");
            urlCell.textContent = videoUrl;

            row.append(thumbnailCell, titleCell, typeCell, transcriptCell, ragCell, publishedCell, durationCell, viewsCell, likesCell, commentsCell, urlCell);
            videoTableBody.appendChild(row);
        });
    }

    const exportOptions = {
        columns: [1, 5, 6, 7, 8, 9, 10],
        modifier: { search: "applied", order: "applied", page: "all" },
        format: {
            body: function(data, row, column, node) {
                if (column === 1) {
                    const link = node.querySelector("a.video-title");
                    return link ? link.textContent.trim() : node.textContent.trim();
                }
                if (column === 4) {
                    return node.querySelector(".status-badge")?.textContent.trim() || node.textContent.trim();
                }
                return node.textContent.trim();
            }
        }
    };

    videoDataTable = $("#videoTable").DataTable({
        pageLength: 10,
        lengthMenu: [10, 25, 50],
        order: [[5, "desc"]],
        columnDefs: [
            { targets: 0, orderable: false, searchable: false },
            { targets: [3, 4], orderable: false },
            { targets: 5, orderSequence: ["desc", "asc"] },
            { targets: 10, visible: false, searchable: false }
        ],
        layout: {
            topStart: ["pageLength", "buttons"],
            topEnd: "search",
            bottomStart: "info",
            bottomEnd: "paging"
        },
        buttons: [
            { extend: "copyHtml5", text: "Copy", exportOptions },
            { extend: "csvHtml5", text: "CSV", filename: "youtube-channel-videos", exportOptions },
            { extend: "excelHtml5", text: "Excel", filename: "youtube-channel-videos", title: "YouTube Channel Videos", exportOptions },
            { extend: "pdfHtml5", text: "PDF", filename: "youtube-channel-videos", title: "YouTube Channel Videos", orientation: "landscape", pageSize: "LETTER", exportOptions },
            { extend: "print", text: "Print", title: "YouTube Channel Videos", exportOptions }
        ],
        orderCellsTop: true
    });

    $("#videoTable thead tr.filter-row th").each(function(columnIndex) {
        $("input, select", this)
            .off("keyup change")
            .on("keyup change", function() {
                const column = videoDataTable.column(columnIndex);
                if (this.tagName === "SELECT" && this.value) {
                    const escaped = $.fn.dataTable.util.escapeRegex(this.value);
                    column.search(`^${escaped}$`, true, false).draw();
                } else {
                    column.search(this.value || "").draw();
                }
            });
    });
    videosSection.classList.remove("hidden");
}


function updateVideoStatusRows(statuses) {
    const byId = new Map((statuses || []).map(item => [String(item.video_id), item]));
    const nodes = videoDataTable ? videoDataTable.rows().nodes().toArray() : Array.from(videoTableBody.querySelectorAll("tr"));
    nodes.forEach(row => {
        const status = byId.get(String(row.dataset.videoId || ""));
        if (!status) return;
        const transcriptCell = row.querySelector(".transcript-status-cell");
        const ragCell = row.querySelector(".rag-status-cell");
        if (transcriptCell) {
            const transcriptLabel = status.transcript_cached ? "Cached" : (status.processing_phase === "transcripts" ? "Retrieving…" : "Not cached");
            transcriptCell.dataset.search = transcriptLabel;
            
            const transcriptTooltip = status.transcript_cached
                ? "Transcript is saved in SQLite"
                : "";
                
            transcriptCell.replaceChildren(
                makeStatusBadge(
                    "transcript",
                    transcriptLabel,
                    status.transcript_cached ? "status-success" : (status.processing_phase ? "status-processing" : "status-missing"),
                    transcriptTooltip
                )
            );
            if (status.transcript_cached) {
                const viewButton = document.createElement("button");
                viewButton.type = "button";
                viewButton.className = "view-transcript-button compact-button";
                viewButton.dataset.videoId = row.dataset.videoId;
                viewButton.textContent = "View Transcript";
                
                transcriptCell.appendChild(viewButton);
            }
        }
        if (ragCell) {
            ragCell.innerHTML = "";
            const label = status.rag_indexed ? "Indexed" : (status.processing_phase === "embedding" ? "Embedding…" : "Not indexed");
            ragCell.dataset.search = label;
            
            const ragTooltip = status.rag_indexed
                ? "Transcript is indexed for chatbot questions"
                : "";

            ragCell.appendChild(
                makeStatusBadge(
                    "rag",
                    label,
                    status.rag_indexed ? "status-success" : (status.processing_phase ? "status-processing" : "status-missing"),
                    ragTooltip
                )
            );
            if (!status.rag_indexed && !status.processing_phase) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "index-video-button compact-button";
                button.dataset.videoId = row.dataset.videoId;
                button.textContent = "Index";
                ragCell.appendChild(button);
            }
        }
        if (videoDataTable) {
            try { videoDataTable.row(row).invalidate("dom"); } catch (error) { /* visible status still updated */ }
        }
    });
}


async function refreshVideoStatuses() {
    if (!activeChannelId) return;
    try {
        const response = await fetch(`/api/channel/${encodeURIComponent(activeChannelId)}/video-statuses`);
        const data = await response.json();
        if (response.ok) updateVideoStatusRows(data.videos);
    } catch (error) {
        console.warn("Unable to refresh video pipeline statuses:", error);
    }
}


function startVideoStatusPolling() {
    if (videoStatusPoller) clearInterval(videoStatusPoller);
    refreshVideoStatuses();
    videoStatusPoller = setInterval(refreshVideoStatuses, 2000);
}


async function analyzeChannel() {
    // Read what the user typed/selected.
    const channelId = channelInput.value.trim();

    // Hide previous results while the new request is running.
    // Also clear the previous channel's question so it does not carry over
    // when a different channel is analyzed.
    activeChannelId = null;
    if (videoStatusPoller) { clearInterval(videoStatusPoller); videoStatusPoller = null; }
    if (indexJobPoller) { clearInterval(indexJobPoller); indexJobPoller = null; }
    activeIndexJobId = null;
    chatSessionId = crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random()}`;
    sessionStorage.setItem("ragChatSessionId", chatSessionId);
    chatMessages.innerHTML = '<div class="chat-bubble assistant">Ask a question about the indexed channel content.</div>';
    questionInput.value = "";
    askQuestionButton.disabled = true;
    errorMessage.classList.add("hidden");
    channelSection.classList.add("hidden");
    analyticsSection.classList.add("hidden");
    videosSection.classList.add("hidden");
    questionSection.classList.add("hidden");
    channelSummarySection.classList.add("hidden");
    channelSummaryResult.classList.add("hidden");
    channelSummaryText.textContent = "";
    channelSummarySourceList.innerHTML = "";
    channelSummaryStatus.textContent = "Index RAG content first, then generate a channel-wide summary.";
    copyChannelSummaryButton.classList.add("hidden");
    transcriptStatus.textContent = "";
    transcriptStatus.className = "transcript-status hidden";

    if (channelId === "") {
        errorMessage.textContent = "Please enter a YouTube channel ID or URL.";
        errorMessage.classList.remove("hidden");
        return;
    }

    loadingMessage.classList.remove("hidden");
    analyzeButton.disabled = true;

    try {
        // No video limit is sent. The backend follows YouTube pagination
        // until it reaches the end of the channel's uploads playlist.

        // Resolves the user's channel ID or URL into a channel ID
        const resolveUrl = `/api/resolve-channel?channel_input=${encodeURIComponent(channelId)}`;
        const resolveResponse = await fetch(resolveUrl);
        const resolveData = await resolveResponse.json();
        
        if (!resolveResponse.ok) {
            throw new Error(resolveData.detail || "Unable to find YouTube channel.");
        }
        
        const resolvedChannelId = resolveData.channel_id;
        
        // Use the resolved channel ID to request the channel analytics
        const url = `/api/channel/${encodeURIComponent(resolvedChannelId)}/analytics`;

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to analyze channel.");
        }

        // One backend request gives us channel, analytics, and videos.
        displayChannel(data.channel, data.retrieved_video_count);
        displayAnalytics(data.analytics);
        displayVideos(data.videos);

        // Hides the loading message after displaying results
        loadingMessage.classList.add("hidden");
        analyzeButton.disabled = false;
        
        // Remember the resolved channel so questions use the same analysis.
        activeChannelId = resolvedChannelId;

        // Shows the Q&A section after channel analysis. RAG indexing is a
        // separate user action, so channel analysis itself does not retrieve
        // transcripts.
        questionSection.classList.remove("hidden");
        channelSummarySection.classList.remove("hidden");
        await refreshRagCoverage();
        startVideoStatusPolling();

    } catch (error) {
        errorMessage.textContent = `Error: ${error.message}`;
        errorMessage.classList.remove("hidden");
    } finally {
        // This always runs whether the request succeeds or fails.
        loadingMessage.classList.add("hidden");
        analyzeButton.disabled = false;
    }
}


// Show how much RAG content is already stored for the active channel.
async function refreshRagCoverage() {
    if (!activeChannelId) return;

    try {
        const response = await fetch(
            `/api/rag/stats?channel_id=${encodeURIComponent(activeChannelId)}`
        );
        const data = await response.json();

        if (!response.ok) return;

        const videoCount = data.video_count || 0;
        const chunkCount = data.chunk_count || 0;

        if (videoCount > 0) {
            displayTranscriptStatus({
                severity: "success",
                message: `RAG already has ${videoCount} indexed video${videoCount === 1 ? "" : "s"} (${chunkCount} transcript chunks) for this channel.`,
            });
            channelSummaryStatus.textContent = `Ready to summarize ${videoCount} indexed video${videoCount === 1 ? "" : "s"}.`;
        } else {
            displayTranscriptStatus({
                severity: "info",
                message: "No transcript content is indexed for RAG yet. Click 'Index RAG Content' when you are ready to retrieve/index transcripts.",
            });
            channelSummaryStatus.textContent = "No indexed videos are available for a channel summary yet. Index RAG content first.";
        }
    } catch (error) {
        // RAG stats are informational; channel analysis should still work if
        // this optional check fails.
        console.warn("Unable to read RAG coverage:", error);
    }
}


// Explicitly retrieve/index missing transcript content for this channel.
async function startIndexJob(videoId = null) {
    if (!activeChannelId) {
        displayTranscriptStatus({ severity: "warning", message: "Analyze a channel before indexing RAG content." });
        return;
    }
    if (activeIndexJobId) {
        displayTranscriptStatus({ severity: "info", message: "An indexing job is already running." });
        return;
    }

    indexRagButton.disabled = true;
    indexRagButton.textContent = "Indexing...";
    cancelIndexButton.classList.remove("hidden");
    indexProgressPanel.classList.remove("hidden");
    indexProgressBar.value = 0;
    indexProgressText.textContent = videoId ? "Starting single-video indexing..." : "Starting channel indexing...";
    indexProgressCount.textContent = "";

    try {
        const response = await fetch("/api/rag/index/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ channel_id: activeChannelId, video_id: videoId })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to start indexing.");
        activeIndexJobId = data.job_id;
        pollIndexJob();
        indexJobPoller = setInterval(pollIndexJob, 1000);
    } catch (error) {
        indexRagButton.disabled = false;
        indexRagButton.textContent = "Index RAG Content";
        cancelIndexButton.classList.add("hidden");
        displayTranscriptStatus({ severity: "warning", message: `RAG indexing error: ${error.message}` });
    }
}


async function indexRagContent() {
    return startIndexJob(null);
}


async function pollIndexJob() {
    if (!activeIndexJobId) return;
    try {
        const response = await fetch(`/api/rag/index/job/${encodeURIComponent(activeIndexJobId)}`);
        const job = await response.json();
        if (!response.ok) throw new Error(job.detail || "Unable to read indexing progress.");

        const total = Number(job.total_video_count || 0);
        const indexed = Number(job.indexed_video_count || job.newly_indexed_count || 0);
        const processed = Number(job.processed_count || 0);
        const already = Number(job.already_indexed_count || 0);
        const completed = Math.min(total, already + processed);
        const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
        indexProgressBar.value = percent;
        indexProgressText.textContent = job.message || "Indexing...";
        indexProgressCount.textContent = total > 0 ? `${completed}/${total} (${percent}%)` : "";
        displayTranscriptStatus({ severity: job.severity || "info", message: job.message || "Indexing..." });
        await refreshVideoStatuses();

        if (["completed", "cancelled", "blocked", "failed"].includes(job.state)) {
            clearInterval(indexJobPoller);
            indexJobPoller = null;
            activeIndexJobId = null;
            indexRagButton.disabled = false;
            indexRagButton.textContent = job.state === "completed" ? "Index RAG Content" : "Resume Indexing";
            cancelIndexButton.classList.add("hidden");
            indexProgressBar.value = job.state === "completed" ? 100 : percent;
            channelSummaryResult.classList.add("hidden");
            channelSummaryText.textContent = "";
            channelSummarySourceList.innerHTML = "";
            copyChannelSummaryButton.classList.add("hidden");
            await refreshRagCoverage();
            await refreshVideoStatuses();
        }
    } catch (error) {
        console.warn("Unable to poll indexing job:", error);
    }
}


async function cancelIndexing() {
    if (!activeIndexJobId) return;
    cancelIndexButton.disabled = true;
    try {
        await fetch(`/api/rag/index/job/${encodeURIComponent(activeIndexJobId)}/cancel`, { method: "POST" });
    } finally {
        cancelIndexButton.disabled = false;
    }
}


// Displays transcript indexing progress or a friendly block warning.
function displayTranscriptStatus(status) {
    if (!status || !status.message) {
        transcriptStatus.textContent = "";
        transcriptStatus.className = "transcript-status hidden";
        return;
    }

    transcriptStatus.textContent = status.message;
    transcriptStatus.className = "transcript-status";

    if (status.severity === "warning") {
        transcriptStatus.classList.add("warning");
    } else if (status.severity === "success") {
        transcriptStatus.classList.add("success");
    }
}


// Handles the Ask Question button
function appendChatBubble(role, text, sources = []) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    const content = document.createElement("div");
    content.textContent = text;
    bubble.appendChild(content);
    if (sources && sources.length) {
        const heading = document.createElement("div");
        heading.textContent = sources.length === 1 ? "Source:" : "Sources:";
        heading.className = "chat-source-heading";
        bubble.appendChild(heading);

        const list = document.createElement("ol");
        list.className = "chat-source-list";

        sources.forEach(source => {
            const item = document.createElement("li");
            const link = document.createElement("a");
            link.href = source.video_url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.title || source.video_id;
            item.appendChild(link);
            if (source.start_time !== null && source.start_time !== undefined) {
                const timestamp = document.createElement("div");
                timestamp.textContent = `Timestamp: ${formatDuration(source.start_time)}`;
                item.appendChild(timestamp);
            }
            list.appendChild(item);
        });
        bubble.appendChild(list);
    }

    // Adds a Copy Answer button to AI responses
    if (role === "assistant") {
        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.className = "chat-copy-button";
        copyButton.textContent = "Copy Answer";

        copyButton.addEventListener("click", async () => {
            let copyText = text;

            if (sources && sources.length) {
                const sourceLines = sources.map((source, index) => {
                    const title = source.title || source.video_id;
                    let sourceText = `${index + 1}. ${title}\n   ${source.video_url}`;

                    if (source.start_time !== null && source.start_time !== undefined) {
                        sourceText += `\n   Timestamp: ${formatDuration(source.start_time)}`;
                    }

                    return sourceText;
                });

                copyText += "\n\nSources:\n" + sourceLines.join("\n");
            }

            try {
                await navigator.clipboard.writeText(copyText);
                copyButton.textContent = "Copied!";
                setTimeout(() => {
                    copyButton.textContent = "Copy Answer";
                }, 2000);
            } catch (error) {
                copyButton.textContent = "Copy failed";
            }
        });

        bubble.appendChild(copyButton);
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


async function askQuestion() {
    const question = questionInput.value.trim();
    if (question === "") return;
    if (!activeChannelId) {
        appendChatBubble("assistant", "Analyze a channel before asking a question.");
        return;
    }

    appendChatBubble("user", question);
    questionInput.value = "";
    askQuestionButton.disabled = true;
    const pending = document.createElement("div");
    pending.className = "chat-bubble assistant pending-answer";
    pending.textContent = "Searching indexed transcript context...";
    chatMessages.appendChild(pending);

    try {
        const response = await fetch("/api/question", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                channel_id: activeChannelId,
                question,
                session_id: chatSessionId
            })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to get an answer.");
        if (data.session_id) {
            chatSessionId = data.session_id;
            sessionStorage.setItem("ragChatSessionId", chatSessionId);
        }
        pending.remove();
        appendChatBubble("assistant", data.answer, data.sources || []);
        displayTranscriptStatus(data.transcript_status);
    } catch (error) {
        pending.remove();
        appendChatBubble("assistant", `Error: ${error.message}`);
    }
}


// FP9: Generate a bounded channel-wide summary from existing Chroma content only.
async function generateChannelSummary() {
    if (!activeChannelId) {
        channelSummaryStatus.textContent = "Analyze a channel before generating a summary.";
        return;
    }

    generateChannelSummaryButton.disabled = true;
    generateChannelSummaryButton.textContent = "Generating...";
    copyChannelSummaryButton.classList.add("hidden");
    channelSummaryResult.classList.add("hidden");
    channelSummaryStatus.textContent = "Summarizing indexed channel content. Larger channels may require several bounded Gemini calls...";

    try {
        const response = await fetch(
            `/api/channel/${encodeURIComponent(activeChannelId)}/summary`,
            { method: "POST" }
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to generate channel summary.");

        const indexedCount = Number(data.indexed_video_count || 0);
        const chunkCount = Number(data.representative_chunk_count || 0);
        channelSummaryText.textContent = data.summary || "No summary was returned.";
        channelSummaryStatus.textContent = `Summarized ${indexedCount} indexed video${indexedCount === 1 ? "" : "s"} using ${chunkCount} representative transcript chunk${chunkCount === 1 ? "" : "s"}.`;

        channelSummarySourceList.innerHTML = "";
        (data.sources || []).forEach(source => {
            const item = document.createElement("li");
            const link = document.createElement("a");
            link.href = source.video_url || `https://www.youtube.com/watch?v=${encodeURIComponent(source.video_id || "")}`;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.title || source.video_id || "Unknown video";
            item.appendChild(link);

            const details = [];
            if (source.published_at) details.push(formatDate(source.published_at));
            if (source.language) details.push(source.language);
            if (details.length) {
                const meta = document.createElement("span");
                meta.className = "channel-summary-source-meta muted";
                meta.textContent = ` — ${details.join(" · ")}`;
                item.appendChild(meta);
            }
            channelSummarySourceList.appendChild(item);
        });

        channelSummaryResult.classList.remove("hidden");
        copyChannelSummaryButton.classList.remove("hidden");
        if (data.generation_warning) {
            channelSummaryStatus.textContent += ` Warning: ${data.generation_warning}`;
        }
    } catch (error) {
        channelSummaryStatus.textContent = `Unable to generate channel summary: ${error.message}`;
    } finally {
        generateChannelSummaryButton.disabled = false;
        generateChannelSummaryButton.textContent = "Generate Channel Summary";
    }
}


async function copyChannelSummary() {
    const text = channelSummaryText.textContent.trim();
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        copyChannelSummaryButton.textContent = "Copied!";
        setTimeout(() => { copyChannelSummaryButton.textContent = "Copy Summary"; }, 2000);
    } catch (error) {
        copyChannelSummaryButton.textContent = "Copy failed";
        setTimeout(() => { copyChannelSummaryButton.textContent = "Copy Summary"; }, 2000);
    }
}


function formatTranscriptSegmentTime(totalSeconds) {
    return formatDuration(Number(totalSeconds || 0));
}


// Loads and displays a specific video's cached transcript in the modal
async function viewVideoTranscript(videoId) {
    const transcriptModal = document.getElementById("transcriptModal");
    const transcriptModalMeta = document.getElementById("transcriptModalMeta");

    transcriptModal.classList.remove("hidden");
    document.body.classList.add("modal-open");
    transcriptModalMeta.textContent = "";
    transcriptModalText.textContent = "Loading cached transcript...";
    copyTranscriptModalButton.disabled = true;

    try {
        const response = await fetch(
            `/api/transcript/${encodeURIComponent(videoId)}`
        );
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to load transcript.");
        }

        const meta = [];

        if (data.language) meta.push(`Language: ${data.language}`);
        if (data.published_at) meta.push(`Published: ${formatDate(data.published_at)}`);
        if (data.duration_seconds !== null && data.duration_seconds !== undefined) {
            meta.push(`Duration: ${formatDuration(data.duration_seconds)}`);
        }

        meta.push(
            data.has_timestamps
                ? "Timestamps: available"
                : "Timestamps: unavailable (legacy cached transcript)"
        );

        transcriptModalMeta.textContent = meta.join(" • ");

        if (data.segments && data.segments.length > 0) {
            transcriptModalText.textContent = data.segments
                .map(segment => {
                    const start = formatTranscriptSegmentTime(segment.start);
                    const endValue = segment.end !== undefined && segment.end !== null
                        ? segment.end
                        : Number(segment.start || 0) + Number(segment.duration || 0);
                    const end = formatTranscriptSegmentTime(endValue);

                    return `[${start}–${end}] ${segment.text || ""}`;
                })
                .join("\n");
        } else {
            transcriptModalText.textContent =
                data.transcript || "This cached transcript is empty.";
        }
        copyTranscriptModalButton.disabled = false;
    } catch (error) {
        transcriptModalText.textContent = `Error: ${error.message}`;
    }
}

generateChannelSummaryButton.addEventListener("click", generateChannelSummary);
copyChannelSummaryButton.addEventListener("click", copyChannelSummary);

// Index transcript content only when the user explicitly requests it.
indexRagButton.addEventListener("click", indexRagContent);
cancelIndexButton.addEventListener("click", cancelIndexing);

videoTableBody.addEventListener("click", event => {
    const indexButton = event.target.closest(".index-video-button");
    if (indexButton) {
        startIndexJob(indexButton.dataset.videoId);
        return;
    }

    const viewButton = event.target.closest(".view-transcript-button");
    if (viewButton) {
        viewVideoTranscript(viewButton.dataset.videoId);
    }
});

openChatButton.addEventListener("click", () => {
    chatModal.classList.remove("hidden");
    document.body.classList.add("modal-open");
    questionInput.focus();
});
closeChatButton.addEventListener("click", () => {
    chatModal.classList.add("hidden");
    document.body.classList.remove("modal-open");
});
chatModal.addEventListener("click", event => {
    if (event.target === chatModal) closeChatButton.click();
});
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !chatModal.classList.contains("hidden")) closeChatButton.click();
});

// Closes the transcript modal
const transcriptModal = document.getElementById("transcriptModal");
const closeTranscriptModal = document.getElementById("closeTranscriptModal");
const closeTranscriptModalButton = document.getElementById("closeTranscriptModalButton");

function closeTranscriptModalWindow() {
    transcriptModal.classList.add("hidden");
    document.body.classList.remove("modal-open");
}

closeTranscriptModal.addEventListener("click", closeTranscriptModalWindow);
closeTranscriptModalButton.addEventListener("click", closeTranscriptModalWindow);

transcriptModal.addEventListener("click", event => {
    if (event.target === transcriptModal) {
        closeTranscriptModalWindow();
    }
});

// Closes the transcript modal with Escape
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !transcriptModal.classList.contains("hidden")) {
        closeTranscriptModalWindow();
    }
});

// Copies the transcript in the modal
copyTranscriptModalButton.addEventListener("click", async () => {
    const transcript = transcriptModalText.innerText.trim();
    if (!transcript) return;

    try {
        await navigator.clipboard.writeText(transcript);
        copyTranscriptModalButton.textContent = "Copied!";

        setTimeout(() => {
            copyTranscriptModalButton.textContent = "Copy Transcript";
        }, 2000);
    } catch (error) {
        copyTranscriptModalButton.textContent = "Copy failed";
    }
});


// Asks the question when the button is clicked. This is Chroma-only and does
// not retrieve missing transcripts.
askQuestionButton.addEventListener("click", askQuestion);

// Asks the question when Enter is pressed
questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter") askQuestion();
});

// Starts a fresh non-persistent conversation in both the browser and Python memory.
newChatButton.addEventListener("click", async () => {
    const oldSessionId = chatSessionId;
    chatSessionId = crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random()}`;
    sessionStorage.setItem("ragChatSessionId", chatSessionId);
    questionInput.value = "";
    chatMessages.innerHTML = '<div class="chat-bubble assistant">New chat started. Ask a question about the indexed channel content.</div>';
    askQuestionButton.disabled = true;
    try { await fetch(`/api/chat/session/${encodeURIComponent(oldSessionId)}`, { method: "DELETE" }); } catch (error) { console.warn(error); }
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

// Clears channel input
clearInputButton.addEventListener("click", () => {
    channelInput.value = "";
    channelInput.focus();
});

// Clears chatbot question input
clearQuestionButton.addEventListener("click", () => {
    questionInput.value = "";
    askQuestionButton.disabled = true;
    questionInput.focus();
});

// Clears the question and answer fields when the page loads
questionInput.value = "";
askQuestionButton.disabled = true;

// Switch between light and dark mode when the theme button is clicked.
themeToggle.addEventListener("click", toggleTheme);

// The small script in index.html already restored the saved class before paint.
// This updates the button label/icon to match that saved theme.
updateThemeToggle();

// Check FastAPI as soon as the page loads.
checkBackendConnection();

// Shows the Back to Top button when scrolling down
const backToTopButton = document.getElementById("backToTopButton");

window.addEventListener("scroll", () => {
    backToTopButton.style.display = window.scrollY > 300 ? "flex" : "none";
});

// Smoothly scrolls back to the top when the button is clicked
backToTopButton.addEventListener("click", () => {
    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
});

// Makes section headings collapsible
document.querySelectorAll(".section-toggle").forEach(button => {
    button.addEventListener("click", () => {
        const contentId = button.getAttribute("aria-controls");
        const content = document.getElementById(contentId);
        const isExpanded = button.getAttribute("aria-expanded") === "true";
        const icon = button.querySelector(".toggle-icon");

        content.classList.toggle("collapsed", isExpanded);
        button.setAttribute("aria-expanded", String(!isExpanded));
        icon.textContent = isExpanded ? "+" : "−";
    });
});

// Text size controls
const decreaseTextSizeButton = document.getElementById("decreaseTextSize");
const resetTextSizeButton = document.getElementById("resetTextSize");
const increaseTextSizeButton = document.getElementById("increaseTextSize");

// Default and allowed text sizes (in pixels)
const defaultTextSize = 16;
const minTextSize = 12;
const maxTextSize = 24;
const textSizeStep = 2;

// Load the saved text size, or use the default
let currentTextSize = Number(localStorage.getItem("textSize")) || defaultTextSize;

// Apply the selected text size
function applyTextSize() {
    document.documentElement.style.fontSize = `${currentTextSize}px`;
    localStorage.setItem("textSize", currentTextSize);
}

// Decrease text size
decreaseTextSizeButton.addEventListener("click", () => {
    if (currentTextSize > minTextSize) {
        currentTextSize = Math.max(minTextSize, currentTextSize - textSizeStep);
        applyTextSize();
    }
});

// Reset text size
resetTextSizeButton.addEventListener("click", () => {
    currentTextSize = defaultTextSize;
    applyTextSize();
});

// Increase text size
increaseTextSizeButton.addEventListener("click", () => {
    if (currentTextSize < maxTextSize) {
        currentTextSize = Math.min(maxTextSize, currentTextSize + textSizeStep);
        applyTextSize();
    }
});

// Apply the saved text size when the page loads
applyTextSize();
