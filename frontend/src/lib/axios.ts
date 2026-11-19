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
    if (error.response) {
      console.warn("API Error:", error.response.status, error.response.data);
    } else if (error.request) {
      console.warn("Network Error: Backend may not be reachable at", error.config?.baseURL || "endpoint");
    } else {
      console.warn("Request Error:", error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
