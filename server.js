const express = require('express');
const app = express();
const PORT = 3000;

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Parse JSON body (using built-in middleware as of Express 4.16+)
app.use(express.json()); 

// Endpoint to handle chat messages
app.post('/chat', (req, res) => {
    const userMessage = req.body.message;

    // Simple bot logic (replace with your bot's logic or call to another service)
    const botReply = `Hello! You said: "${userMessage}"`;

    res.json({ reply: botReply });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
