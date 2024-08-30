const  express =require('express');
const { getGrid } = require('../controllers/grid')

const router = express.Router();

router.route('/').get(getGrid);

module.exports =router;