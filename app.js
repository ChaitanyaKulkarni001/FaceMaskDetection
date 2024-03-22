
const express = require('express')
const app = express();
const path = require('path');

app.use(express.static(path.join(__dirname, '\web'))); // Path is C:\Hackathon24\FaceMaskDetection\web
// const express = require('express');
const { spawn } = require('child_process');

// const app = express();
const port = 3008;

// Define route for the root URL
app.get('/', (req, res) => {
    // Send the index.html file
    const indexPath = path.join(__dirname, 'web', 'index.html');
    
    // Send the index.html file
    res.sendFile(indexPath)
});

// Define route for executing the command
app.get('/execute', (req, res) => {
   var data2send;
   const python=spawn('python',['C:\\Hackathon24\\Face-Mask-Detection-master\\Face-Mask-Detection-master\\only_face_detection.py']);
   python.stdout.on('data',function(data){
    data2send = data.toString();

   });
   python.stderr.on('data',data=>{
    console.error(`stderr:${data}`)
   })

   python.on('exit',(code)=>{
    console.log(`child process exited with code ${code},${data2send}`)
    // express.response.sendFile(`${__dirname}/public/res`)
   })
});
 
// Start the server
app.listen(port, () => {
    console.log(`Server is listening at http://localhost:${port}`);
});


