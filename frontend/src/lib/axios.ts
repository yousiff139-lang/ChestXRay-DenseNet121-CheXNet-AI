import axios from "axios";

// Create a custom axios instance with default configuration
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
  // timeout: 30000, // 30 seconds
});

// Add request interceptor for authentication and path normalization
apiClient.interceptors.request.use(
  (config) => {
    // Normalize path to prevent duplicate /api/v1 if baseURL already includes it
    if (config.baseURL?.endsWith("/api/v1") && config.url?.startsWith("/api/v1")) {
      config.url = config.url.replace(/^\/api\/v1/, "");
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors here
    if (error.response) {
      // Server responded with a status code outside the 2xx range
      console.error("API Error:", error.response.data);
    } else if (error.request) {
      // Request was made but no response was received
      console.error("Network Error:", error.message);
    } else {
      // Something happened in setting up the request
      console.error("Request Error:", error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
