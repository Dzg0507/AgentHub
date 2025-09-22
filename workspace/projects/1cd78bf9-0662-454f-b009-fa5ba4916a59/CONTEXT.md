# Project Context

## Original Prompt
Create a simple todo app

## Enhanced Prompt
Create a full-stack TODO application with the following specifications:

**I. Frontend:**

* **Technology:** React 18 with TypeScript.  The application should be responsive and adhere to modern web design principles.  Consider using a UI framework like Material-UI or Chakra UI for consistent styling and component reusability.
* **Features:**
    * Add new TODO items with a description and optional due date.
    * Mark TODO items as complete.
    * Edit existing TODO items.
    * Delete TODO items.
    * Filter TODO items by status (completed/incomplete).
    * Optionally, implement sorting capabilities (e.g., by due date, creation date).
    * Display the number of completed and incomplete TODO items.
* **State Management:** Utilize a state management library like Redux Toolkit or Zustand for efficient management of application state.  Justify your choice in the code comments.
* **Testing:** Implement comprehensive unit and integration tests using Jest and React Testing Library. Aim for high test coverage (at least 80%).

**II. Backend:**

* **Technology:** Node.js with Express.js.  The backend will provide a RESTful API to interact with the frontend.
* **Database:** Use PostgreSQL with a well-defined schema. Create a migration system (e.g., using `sequelize-cli` or `typeorm`) to manage database schema changes.  The schema should include columns for ID (UUID), description (text), due date (timestamp with time zone), and completed status (boolean).  Consider adding user authentication later.
* **API Design:**  Design a RESTful API with endpoints for creating, reading, updating, and deleting TODO items.  Include proper HTTP status codes and error handling.  Use descriptive endpoint names and follow RESTful best practices.
* **Error Handling:** Implement robust error handling throughout the backend, including logging and appropriate HTTP responses for different error scenarios.
* **Testing:** Write unit and integration tests for the backend API using a testing framework like Jest or Supertest.

**III.  Authentication (Optional but Recommended):**

* Implement a JWT-based authentication system to secure the API.  Consider using Passport.js for authentication middleware.  This feature should be designed in a modular way so it can be easily enabled or disabled.

**IV. Deployment:**

* **Containerization:**  Package the application using Docker for ease of deployment and consistent environment.  Create a Dockerfile for both the frontend and backend.
* **CI/CD:**  Set up a CI/CD pipeline (e.g., using GitHub Actions, GitLab CI, or Jenkins) to automate the build, testing, and deployment process.
* **Deployment Platform:** Consider deploying to a cloud platform like Heroku, AWS, Google Cloud, or Netlify.

**V. Security:**

* **Input Validation:** Implement input validation to prevent vulnerabilities such as SQL injection and cross-site scripting (XSS).  Use parameterized queries or ORMs to avoid SQL injection.
* **CORS:** Configure CORS properly to allow requests from the frontend to the backend.
* **Authentication:** If implemented, ensure the JWT authentication is secure and follows best practices.

**VI.  Performance:**

* Consider database indexing to optimize query performance.
* Implement appropriate caching strategies if needed (e.g., Redis).

**VII.  File Structure & Organization:**

The project should follow a clear and organized file structure.  Suggest a file structure before beginning development.  Examples include separating frontend and backend code into distinct folders, and organizing components and API routes logically.


**VIII. Testing and Validation:**

- Frontend:  Unit tests for components, integration tests for API interactions. Aim for >80% test coverage.
- Backend: Unit tests for API routes, integration tests for database interactions. Aim for >80% test coverage.
- End-to-End Testing: Consider adding E2E tests using Cypress or similar tools to test the entire application flow.

**IX.  Documentation:**

Provide clear documentation for the API, including endpoint specifications, request/response formats, and error codes. Include a README file with setup instructions and usage examples.  Consider using Swagger or OpenAPI for API documentation.

## Language
python

## Guidance Level
minimal
