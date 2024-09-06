const  express =require('express');
const { postUpdate } = require('../controllers/update')

const router = express.Router();

router.route('/update').post(postUpdate);

module.exports =router;