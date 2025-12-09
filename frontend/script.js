(function() {
  // Global variable to hold the Chart.js instance
  let myChartInstance = null;
  const TARGET_FLOW_ID = 'c2739ce8-e210-40aa-850e-0ebb29a26be7';
  console.log('[Script.js] Loaded successfully, event listener registered, target flowId:', TARGET_FLOW_ID);

  // Listen for message addition events (triggered by AI responses)
  window.addEventListener('langflow:messages:added', (event) => {
    console.log('[Script.js] Received add event', {
      flowId: event.detail.flow_id,
      isAiMessage: !event.detail.newMessage.isSend
    });

    const { newMessage, flow_id } = event.detail;
    // Pass only necessary parameters to the handler function
    handleNewMessage(newMessage, flow_id);
  });

  // Extract JSON fragment from text (core fix logic)
  function extractJsonFromText(text) {
    // Match JSON block from first { to last } (supports multi-line/single-line)
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON block found in message');
    }
    return jsonMatch[0];
  }

  // Handle new messages (added flow_id parameter to match target components)
  function handleNewMessage(message, flow_id) {
    console.log('[Script.js] Processing message', {
      flowId: flow_id,
      isSend: message.isSend,
      content: message.message
    });

    // Core validation: 1. Is AI response (!message.isSend) 2. flow_id matches target
    const isTargetChat = flow_id === TARGET_FLOW_ID;
    if (!message.isSend && isTargetChat) {
      const messageText = message.message;

      try {
        // Step 1: Extract JSON fragment (core fix)
        const jsonText = extractJsonFromText(messageText);
        // Step 2: Parse JSON
        const parsedMessage = JSON.parse(jsonText);

        if (!parsedMessage) throw new Error('JSON parsed as empty');

        // Check required fields for spectrum data
        const requiredFields = ['type', 'light_start_lambda', 'light_stop_lambda', 'light_points', 'intensities'];
        const missingFields = requiredFields.filter(field => !(field in parsedMessage));
        if (missingFields.length > 0) {
          throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
        }

        // Render the spectrum
        if (parsedMessage.type === "spectrum" && Array.isArray(parsedMessage.intensities)) {
          try {
            renderSpectrum(parsedMessage);
            console.log('[Script.js] renderSpectrum executed successfully');
          } catch (renderErr) {
            console.error('[Script.js] Failed to render chart:', renderErr.message, renderErr.stack);
          }
        } else if (parsedMessage.type === "structure") {
          console.log("Received structure data:", parsedMessage);
          // Can add processing logic for structure data
          document.getElementById('no-data-message').textContent = 'Received structure data, visualization not supported yet';
          document.getElementById('no-data-message').style.display = 'block';
          if (myChartInstance) myChartInstance.destroy(); // Clear old chart if new data type
        }
      } catch (e) {
        console.error('Failed to parse spectrum data:', e.message, 'Original message:', messageText);
        // Optimized error prompt (more precise)
        const errorMsg = e.message.includes('No valid JSON block')
          ? 'No valid spectrum data (invalid JSON format) found in AI response, unable to generate chart'
          : 'AI response format error, unable to generate spectrum chart';
        document.getElementById('no-data-message').textContent = errorMsg;
        document.getElementById('no-data-message').style.display = 'block';
        if (myChartInstance) myChartInstance.destroy(); // Clear old chart on error
      }
    }
  }

  // Function to render spectrum (unchanged, only optimized log output)
  function renderSpectrum(spectrumData) {
    const {
      intensities = [], // Default value: empty array
      light_start_lambda = 400, // Default value: 400nm
      light_stop_lambda = 800, // Default value: 800nm
      light_points = intensities.length, // Default value: length of intensities
      title = "Predicted Optical Spectrum",
      x_label = "Wavelength (nm)",
      y_label = "Intensity"
    } = spectrumData;

    // Filter invalid data (intensity must be a number)
    const validIntensities = intensities.filter(val => typeof val === 'number');
    const validPoints = Math.min(light_points, validIntensities.length);

    if (validPoints === 0) {
      document.getElementById('no-data-message').textContent = 'No valid spectrum data, unable to render';
      document.getElementById('no-data-message').style.display = 'block';
      // Destroy existing chart
      if (myChartInstance) myChartInstance.destroy();
      return;
    }

    const trimmedIntensities = validIntensities.slice(0, validPoints);
    const canvas = document.getElementById('myChart');
    console.log('[Script.js] Chart container status:', { canvasExists: !!canvas });

    if (!canvas) {
      throw new Error('Canvas element with id "myChart" not found');
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get canvas 2D context');
    }

    // Hide prompt message
    document.getElementById('no-data-message').style.display = 'none';

    // Calculate wavelength array: optimize precision to avoid floating-point errors
    const wavelengths = [];
    if (validPoints > 1) {
      const lambdaStep = (light_stop_lambda - light_start_lambda) / (validPoints - 1);
      for (let i = 0; i < validPoints; i++) {
        wavelengths.push(parseFloat((light_start_lambda + i * lambdaStep).toFixed(2)));
      }
    } else {
      wavelengths.push(light_start_lambda);
    }

    // Destroy existing chart (to avoid duplicate rendering)
    if (myChartInstance) {
      console.log('[Script.js] Destroy existing chart instance');
      myChartInstance.destroy();
    }

    // Create new chart
    myChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: wavelengths,
        datasets: [{
          label: title,
          data: trimmedIntensities,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)', // Area under the line
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 0,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1000,
          easing: 'easeOutQuart'
        },
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: title,
            font: { size: 16, weight: 'bold' },
            padding: { bottom: 20 }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            padding: 10,
            callbacks: {
              title: (context) => `${x_label}: ${context[0].label} nm`,
              label: (context) => `${y_label}: ${context.raw.toFixed(4)}`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: y_label, font: { size: 14 } },
            beginAtZero: true,
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: { precision: 4 }
          },
          x: {
            title: { display: true, text: x_label, font: { size: 14 } },
            grid: { display: false },
            ticks: { autoSkip: true, maxTicksLimit: 10 }
          }
        }
      }
    });
  }
})();