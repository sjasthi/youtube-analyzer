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

// Dark-mode toggle elements.
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");

// Gets references to the RAG question and answer elements
const questionSection = document.getElementById("questionSection");
const questionInput = document.getElementById("questionInput");
const askQuestionButton = document.getElementById("askQuestionButton");
const indexRagButton = document.getElementById("indexRagButton");
const answerText = document.getElementById("answerText");
const answerSources = document.getElementById("answerSources");
const transcriptStatus = document.getElementById("transcriptStatus");
const clearQuestionButton = document.getElementById("clearQuestionButton");
const answerLanguage = document.getElementById("answerLanguage");
const exampleQuestions = document.querySelectorAll(".example-question");

// Cache-only transcript viewer elements.
const transcriptViewerSection = document.getElementById("transcriptViewerSection");
const transcriptVideoSelect = document.getElementById("transcriptVideoSelect");
const loadTranscriptButton = document.getElementById("loadTranscriptButton");
const transcriptViewerMeta = document.getElementById("transcriptViewerMeta");
const transcriptViewerText = document.getElementById("transcriptViewerText");

// Default messages for the question and answer section
const defaultAnswer = "The answer will appear here.";

// Store the current Chart.js object so it can be replaced on the next search.
let uploadTrendChart = null;

// Store the current DataTables object so it can be replace on the next search.
let videoDataTable = null;

// Resolved UC channel ID used by the RAG question endpoint.
let activeChannelId = null;

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

        videoLink.href = `https://www.youtube.com/watch?v=${encodeURIComponent(video.video_id)}`;
        videoLink.target = "_blank";
        videoLink.rel = "noopener noreferrer";
        videoLink.className = "video-title";
        videoLink.textContent = video.title;

        mostViewedTitle.replaceChildren(videoLink);
        mostViewedValue.textContent = `${formatNumber(video.view_count)} views`;
    }

    if (analytics.most_liked_video) {
        const video = analytics.most_liked_video;
        const videoLink = document.createElement("a");

        videoLink.href = `https://www.youtube.com/watch?v=${encodeURIComponent(video.video_id)}`;
        videoLink.target = "_blank";
        videoLink.rel = "noopener noreferrer";
        videoLink.className = "video-title";
        videoLink.textContent = video.title;

        mostLikedTitle.replaceChildren(videoLink);
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


function displayVideos(videos) {
    // If DataTables was already created from a previous search,
    // destroy it before changing the table rows.
    if (videoDataTable) {
        videoDataTable.destroy();
        videoDataTable = null;
    }

    // Clear rows from the previous channel analysis.
    videoTableBody.innerHTML = "";

    if (videos.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        // There are 8 table columns: thumbnail, video details, statistics,
        // and a hidden YouTube URL used by the export buttons.
        cell.colSpan = 8;
        cell.textContent = "No public videos were returned.";
        row.appendChild(cell);
        videoTableBody.appendChild(row);
    } else {
        // Create one table row for each returned video.
        videos.forEach(video => {
            const row = document.createElement("tr");

            const titleCell = document.createElement("td");
            titleCell.className = "video-cell";

            // Build the normal YouTube watch URL from the video ID.
            const videoUrl = `https://www.youtube.com/watch?v=${video.video_id}`;

            // Show the YouTube thumbnail as a clickable image. The thumbnail
            // is loaded directly from the URL already returned by the backend.
            const thumbnailCell = document.createElement("td");
            thumbnailCell.className = "video-thumbnail-cell";

            if (video.thumbnail_url) {
                const thumbnailLink = document.createElement("a");
                thumbnailLink.href = videoUrl;
                thumbnailLink.target = "_blank";
                thumbnailLink.rel = "noopener noreferrer";
                thumbnailLink.setAttribute("aria-label", `Open ${video.title || "video"} on YouTube`);

                const thumbnail = document.createElement("img");
                thumbnail.className = "video-thumbnail";
                thumbnail.src = video.thumbnail_url;
                thumbnail.alt = `${video.title || "Video"} thumbnail`;
                thumbnail.loading = "lazy";

                thumbnailLink.appendChild(thumbnail);
                thumbnailCell.appendChild(thumbnailLink);
            } else {
                const unavailable = document.createElement("span");
                unavailable.className = "thumbnail-unavailable";
                unavailable.textContent = "No image";
                thumbnailCell.appendChild(unavailable);
            }

            // Make the video title clickable. It opens YouTube in a new tab.
            const title = document.createElement("a");
            title.className = "video-title";
            title.textContent = video.title || "Untitled";
            title.href = videoUrl;
            title.target = "_blank";
            title.rel = "noopener noreferrer";

            // Keep the video ID visible below the title.
            const id = document.createElement("span");
            id.className = "video-id";
            id.textContent = video.video_id;

            titleCell.appendChild(title);
            titleCell.appendChild(id);

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

            // This eighth column is hidden on the webpage but included
            // in exported CSV, Excel, PDF, Copy, and Print output.
            const urlCell = document.createElement("td");
            urlCell.textContent = videoUrl;

            row.append(
                thumbnailCell,
                titleCell,
                publishedCell,
                durationCell,
                viewsCell,
                likesCell,
                commentsCell,
                urlCell
            );

            videoTableBody.appendChild(row);
        });
    }

    // Export only rows that match the current DataTables search/filter.
    // page: "all" means export every matching row, not just the current page.
    const exportOptions = {
        // Skip column 0 because image thumbnails do not belong in text exports.
        columns: [1, 2, 3, 4, 5, 6, 7],
        modifier: {
            search: "applied",
            order: "applied",
            page: "all"
        },
        format: {
            body: function(data, row, column, node) {
                // Column 1 is the visible Video cell (column 0 is the thumbnail).
                // Export only the readable title because the full URL is exported
                // in its own hidden column.
                if (column === 1) {
                    const link = node.querySelector("a.video-title");
                    return link ? link.textContent.trim() : node.textContent.trim();
                }

                return node.textContent.trim();
            }
        }
    };

    // Create the DataTable with filtering, sorting, pagination, and exports.
    videoDataTable = $("#videoTable").DataTable({
        pageLength: 10,
        lengthMenu: [10, 25, 50],

        // Published defaults to newest first. Thumbnail is column 0, so
        // the Published column is index 2.
        order: [[2, "desc"]],

        columnDefs: [
            {
                // Thumbnail is visual only; it should not affect search or sorting.
                targets: 0,
                orderable: false,
                searchable: false
            },
            {
                // Column 2 = Published.
                targets: 2,
                // Only switch between newest and oldest.
                orderSequence: ["desc", "asc"]
            },
            {
                // Column 7 contains the full YouTube URL.
                // Keep it out of the visible table, but available for export.
                targets: 7,
                visible: false,
                searchable: false
            }
        ],

        // Keep the normal page-length control and add export buttons.
        layout: {
            topStart: ["pageLength", "buttons"],
            topEnd: "search",
            bottomStart: "info",
            bottomEnd: "paging"
        },

        buttons: [
            {
                extend: "copyHtml5",
                text: "Copy",
                exportOptions: exportOptions
            },
            {
                extend: "csvHtml5",
                text: "CSV",
                filename: "youtube-channel-videos",
                exportOptions: exportOptions
            },
            {
                extend: "excelHtml5",
                text: "Excel",
                filename: "youtube-channel-videos",
                title: "YouTube Channel Videos",
                exportOptions: exportOptions
            },
            {
                extend: "pdfHtml5",
                text: "PDF",
                filename: "youtube-channel-videos",
                title: "YouTube Channel Videos",
                orientation: "landscape",
                pageSize: "LETTER",
                exportOptions: exportOptions
            },
            {
                extend: "print",
                text: "Print",
                title: "YouTube Channel Videos",
                exportOptions: exportOptions
            }
        ],

        // Prevent the second header row from becoming another sortable heading row.
        orderCellsTop: true
    });

    // Connect the individual column filters.
    // Thumbnail and hidden URL columns intentionally have no filter input.
    $("#videoTable thead tr.filter-row th").each(function (columnIndex) {
        $("input", this)
            .off("keyup change")
            .on("keyup change", function() {
                videoDataTable
                    .column(columnIndex)
                    .search(this.value)
                    .draw();
            });
    });

    // Show the video section.
    videosSection.classList.remove("hidden");
}


async function analyzeChannel() {
    // Read what the user typed/selected.
    const channelId = channelInput.value.trim();

    // Hide previous results while the new request is running.
    // Also clear the previous channel's question so it does not carry over
    // when a different channel is analyzed.
    activeChannelId = null;
    questionInput.value = "";
    answerText.textContent = defaultAnswer;
    answerSources.innerHTML = "<li>Sources will appear here after you ask a question.</li>";
    askQuestionButton.disabled = true;
    errorMessage.classList.add("hidden");
    channelSection.classList.add("hidden");
    analyticsSection.classList.add("hidden");
    videosSection.classList.add("hidden");
    questionSection.classList.add("hidden");
    transcriptViewerSection.classList.add("hidden");
    transcriptVideoSelect.innerHTML = '<option value="">No cached transcripts loaded</option>';
    loadTranscriptButton.disabled = true;
    transcriptViewerMeta.textContent = "";
    transcriptViewerText.textContent = "Select a cached transcript to view it here.";
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

        // Remember the resolved channel so questions use the same analysis.
        activeChannelId = resolvedChannelId;

        // Shows the Q&A section after channel analysis. RAG indexing is a
        // separate user action, so channel analysis itself does not retrieve
        // transcripts.
        questionSection.classList.remove("hidden");
        transcriptViewerSection.classList.remove("hidden");
        await refreshRagCoverage();
        await refreshTranscriptViewer();

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
        } else {
            displayTranscriptStatus({
                severity: "info",
                message: "No transcript content is indexed for RAG yet. Click 'Index RAG Content' when you are ready to retrieve/index transcripts.",
            });
        }
    } catch (error) {
        // RAG stats are informational; channel analysis should still work if
        // this optional check fails.
        console.warn("Unable to read RAG coverage:", error);
    }
}


// Explicitly retrieve/index missing transcript content for this channel.
async function indexRagContent() {
    if (!activeChannelId) {
        displayTranscriptStatus({
            severity: "warning",
            message: "Analyze a channel before indexing RAG content.",
        });
        return;
    }

    const originalText = indexRagButton.textContent;
    indexRagButton.disabled = true;
    indexRagButton.textContent = "Indexing...";

    displayTranscriptStatus({
        severity: "info",
        message: "Indexing missing transcript content. Cached SQLite/Chroma data will be reused first.",
    });

    try {
        const response = await fetch("/api/rag/index", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                channel_id: activeChannelId,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to index RAG content.");
        }

        displayTranscriptStatus(data.transcript_status);
        await refreshTranscriptViewer();
    } catch (error) {
        displayTranscriptStatus({
            severity: "warning",
            message: `RAG indexing error: ${error.message}`,
        });
    } finally {
        indexRagButton.disabled = false;
        indexRagButton.textContent = originalText;
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
async function askQuestion() {
    const question = questionInput.value.trim();

    // Checks for an empty question
    if (question === "") {
        answerText.textContent = "Please enter a question.";
        return;
    }

    // The question must belong to a channel that was successfully analyzed.
    if (!activeChannelId) {
        answerText.textContent = "Analyze a channel before asking a question.";
        return;
    }
    
    answerText.textContent = "Searching indexed transcript context and generating an answer...";
    transcriptStatus.textContent = "Searching existing Chroma transcript content...";
    transcriptStatus.className = "transcript-status";
    
    try {
        const response = await fetch("/api/question", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            
            body: JSON.stringify({
                channel_id: activeChannelId,
                question: question,
                answer_language: answerLanguage.value.trim() || "auto",
            }),
        });
        
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to get an answer.");
        }

        answerText.textContent = data.answer;
        displayTranscriptStatus(data.transcript_status);
        
        // Displays the source videos used to answer the question
        answerSources.innerHTML = "";
        
        // Creates links to the source videos, or displays a message if none are found
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach((source) => {
                const listItem = document.createElement("li");
                const link = document.createElement("a");

                link.href = source.video_url;
                link.textContent = source.title;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.className = "video-title";

                listItem.appendChild(link);

                const details = [];
                if (source.start_time !== null && source.start_time !== undefined) {
                    details.push(`timestamp ${formatDuration(source.start_time)}`);
                }
                if (source.language) {
                    details.push(`language ${source.language}`);
                }
                if (source.published_at) {
                    details.push(`published ${formatDate(source.published_at)}`);
                }
                if (source.duration_seconds !== null && source.duration_seconds !== undefined) {
                    details.push(`duration ${formatDuration(source.duration_seconds)}`);
                }

                if (details.length > 0) {
                    const metadata = document.createElement("span");
                    metadata.className = "source-meta";
                    metadata.textContent = ` — ${details.join(" • ")}`;
                    listItem.appendChild(metadata);
                }

                answerSources.appendChild(listItem);
            });
        } else {
            answerSources.innerHTML = "<li>No source videos were found.</li>";
        }

    } catch (error) {
        answerText.textContent = `Error: ${error.message}`;
        answerSources.innerHTML = "<li>Sources are unavailable.</li>";
        transcriptStatus.textContent = "";
        transcriptStatus.className = "transcript-status hidden";
    }
}

// Load the list of transcripts that already exist in SQLite. This endpoint is
// cache-only and will never trigger youtube-transcript-api.
async function refreshTranscriptViewer() {
    if (!activeChannelId) return;

    try {
        const response = await fetch(
            `/api/channel/${encodeURIComponent(activeChannelId)}/cached-transcripts`
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to read cached transcripts.");

        transcriptVideoSelect.innerHTML = "";
        const transcripts = data.transcripts || [];

        if (transcripts.length === 0) {
            transcriptVideoSelect.innerHTML = '<option value="">No cached transcripts yet</option>';
            loadTranscriptButton.disabled = true;
            transcriptViewerMeta.textContent = "";
            transcriptViewerText.textContent = "Index RAG content or run youtube_transcriptor.py to create cached transcripts.";
            return;
        }

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = `Choose a transcript (${transcripts.length} cached)`;
        transcriptVideoSelect.appendChild(placeholder);

        transcripts.forEach(item => {
            const option = document.createElement("option");
            option.value = item.video_id;
            const language = item.language ? ` [${item.language}]` : "";
            option.textContent = `${item.title}${language}`;
            transcriptVideoSelect.appendChild(option);
        });

        loadTranscriptButton.disabled = false;
        transcriptViewerMeta.textContent = `${transcripts.length} transcript${transcripts.length === 1 ? "" : "s"} available in SQLite.`;
        transcriptViewerText.textContent = "Select a cached transcript to view it here.";
    } catch (error) {
        loadTranscriptButton.disabled = true;
        transcriptViewerMeta.textContent = `Unable to load cached transcripts: ${error.message}`;
    }
}

function formatTranscriptSegmentTime(totalSeconds) {
    return formatDuration(Number(totalSeconds || 0));
}

async function viewSelectedTranscript() {
    const videoId = transcriptVideoSelect.value;
    if (!videoId) {
        transcriptViewerText.textContent = "Choose a cached transcript first.";
        return;
    }

    loadTranscriptButton.disabled = true;
    transcriptViewerText.textContent = "Loading cached transcript...";

    try {
        const response = await fetch(`/api/transcript/${encodeURIComponent(videoId)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to load transcript.");

        const meta = [];
        if (data.language) meta.push(`Language: ${data.language}`);
        if (data.published_at) meta.push(`Published: ${formatDate(data.published_at)}`);
        if (data.duration_seconds !== null && data.duration_seconds !== undefined) {
            meta.push(`Duration: ${formatDuration(data.duration_seconds)}`);
        }
        meta.push(data.has_timestamps ? "Timestamps: available" : "Timestamps: unavailable (legacy cached transcript)");
        transcriptViewerMeta.textContent = meta.join(" • ");

        if (data.segments && data.segments.length > 0) {
            transcriptViewerText.textContent = data.segments
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
            transcriptViewerText.textContent = data.transcript || "This cached transcript is empty.";
        }
    } catch (error) {
        transcriptViewerText.textContent = `Error: ${error.message}`;
    } finally {
        loadTranscriptButton.disabled = false;
    }
}

transcriptVideoSelect.addEventListener("change", () => {
    if (transcriptVideoSelect.value) viewSelectedTranscript();
});
loadTranscriptButton.addEventListener("click", viewSelectedTranscript);

// Index transcript content only when the user explicitly requests it.
indexRagButton.addEventListener("click", indexRagContent);

// Asks the question when the button is clicked. This is Chroma-only and does
// not retrieve missing transcripts.
askQuestionButton.addEventListener("click", askQuestion);

// Asks the question when Enter is pressed
questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter") askQuestion();
});

// Clears the question and answer
clearQuestionButton.addEventListener("click", () => {
    questionInput.value = "";
    answerText.textContent = defaultAnswer;
    transcriptStatus.textContent = "";
    transcriptStatus.className = "transcript-status hidden";
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
