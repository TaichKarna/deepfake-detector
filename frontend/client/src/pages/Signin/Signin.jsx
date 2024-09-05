import { useForm } from '@mantine/form';
import {
  TextInput,
  PasswordInput,
  Text,
  Paper,
  Group,
  Button,
  Divider,
  Anchor,
  Stack,
} from '@mantine/core';
import { useMutation } from '@tanstack/react-query';
import { GoogleButton } from '../../components/Buttons/GoogleButton';
import { GithubButton } from '../../components/Buttons/GithubButton';
import classes from './Signin.module.css'
import { signIn } from '../../api/auth';
import { useNavigate } from 'react-router-dom';
import useStore from '../../store/store';


export default function Signin(props) {
  const navigate = useNavigate();
  const addUser = useStore(state => state.addUser);
  const form = useForm({
    initialValues: {
      email: '',
      password: '',
    },

    validate: {
      email: (val) => (/^\S+@\S+$/.test(val) ? null : 'Invalid email'),
      password: (val) => (val.length <= 6 ? 'Password should include at least 6 characters' : null),
    },
  });


  const mutation = useMutation({
    mutationFn: signIn,
    onSuccess: async (data) => {
      navigate('/');
      console.log(data)
      addUser(data.user);
    }
  });

  const handleSubmit = async(e) => {
    e.preventDefault();
    mutation.mutate(form.getValues())
  }
  
  return (
    <div className={classes.container}>
      <Paper radius="md" p="xl" withBorder {...props}>
      <Text size="lg" fw={500}>
        Welcome to SensorGrid, login with
      </Text>

      <Group grow mb="md" mt="md">
        <GoogleButton radius="xl">Google</GoogleButton>
        <GithubButton radius="xl">Github</GithubButton>
      </Group>

      <Divider label="Or continue with email" labelPosition="center" my="lg" />

      <form onSubmit={handleSubmit}>
        <Stack>
          <TextInput
            required
            label="Email"
            placeholder="hello@mantine.dev"
            value={form.values.email}
            onChange={(event) => form.setFieldValue('email', event.currentTarget.value)}
            error={form.errors.email && 'Invalid email'}
            radius="md"
          />
          <PasswordInput
            required
            label="Password"
            placeholder="Your password"
            value={form.values.password}
            onChange={(event) => form.setFieldValue('password', event.currentTarget.value)}
            error={form.errors.password && 'Password should include at least 6 characters'}
            radius="md"
          />

        </Stack>
        {
          mutation.isError && (
            <Text  c='#a90003' pt='sm' style={{textAlign: 'center'}}>{mutation.error.message}</Text>
          )
        }
        <Group justify="space-between" mt="xl">
          <Anchor component="" type="button" c="dimmed"  size="xs" >
            {
               `Don't have an account? Sign Up`
            }
          </Anchor>
          <Button type="submit" radius="xl">
            Submit
          </Button>
        </Group>
      </form>
    </Paper>
    </div>
  );
}