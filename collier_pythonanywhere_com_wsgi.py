import sys

# # add your project directory to the sys.path
# project_home = u'/home/collier/mysite'
# if project_home not in sys.path:
#     sys.path = [project_home] + sys.path

# # need to pass the flask app as "application" for WSGI to work
# # for a dash app, that is at app.server
# # see https://plot.ly/dash/deployment
# # from stat_calc_app import app

# from stat_calc_app import app

# application = app.server


import sys

path = '/home/collier/mysite'
if path not in sys.path:
   sys.path.append(path)

from stat_calc_app import stat_calc_app
application = stat_calc_app.server